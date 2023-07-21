import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import TDGCN.Data08.transformer.Constants as Constants
from TDGCN.Data08.transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """
    Get the non-padding positions.
    获得非填充位置。
    """
    """
    assert 断言函数作用:
    断言函数是对表达式布尔值的判断，要求表达式计算值必须为真。可用于自动调试。如果表达式为假，触发异常；如果表达式为真，不执行任何操作。
    """
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """
    For masking out the padding part of key sequence.
    用于屏蔽键序列的填充部分。
    """

    # expand to fit the shape of key query attention matrix 展开以符合关键查询注意矩阵的形状
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """
    For masking out the subsequent info, i.e., masked self-attention.
    用来屏蔽后续信息，也就是屏蔽自我注意。
    """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    具有自我注意机制的编码模型。
    """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding 位置矢量，用于时间编码
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))
        # event type embedding 事件类型嵌入，公式（1）
        # 一共有num_types + 1个词，每个词用d_model(512)维向量表征，对应的权重就是一个76×512的矩阵
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        通过蒙面自我注意编码事件序列。
        """

        # prepare attention masks 准备注意面具
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        # slf_attn_mask是我们不能看的地方，即，未来和填充
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """
    Prediction of next event type.
    预测下一个事件类型。
    """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        # 正态分布
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    可选的循环层。这是因为在Transformer之上添加循环层有助于语言建模。
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """
    A sequence to sequence model with attention mechanism.
    具有注意机制的序列对序列模型。
    """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar 将隐藏向量转换为标量 512X2
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference 参数用于时差的权重
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function softplus函数的参数
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp 预测下一个时间戳
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type 下一个事件类型的预测
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        返回隐藏的表示和预测。
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        """
          get_non_pad_mask:Get the non-padding positions.获得非填充位置。
        """
        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

