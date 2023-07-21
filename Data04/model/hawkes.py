import numpy as np
import time
import torch

import TDGCN.Data08.transformer.Constants as Constants
import TDGCN.Data08.Utils as Utils

from TDGCN.Data08.preprocess.Dataset import get_dataloader
from tqdm import tqdm

def prepare_dataloader(A, opt):
    """ Load data and prepare dataloader. """

    num_types = 3
    hawkes_batch = 1

    def load_data(A):
        h = 1e-3

        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0

        "生成事件矩阵E"
        Event_list = []
        for i in range(1, len(A)):
            Event_list.append(A[i] - A[i - 1])

        len_Event_list = len(Event_list)
        print("Event_list_len:", len(Event_list))

        "将事件矩阵处理成{'time_since_start': , 'time_since_last_event': , 'type_event': }格式"
        data = []
        for i in range(opt.num_point):
            for j in range(opt.num_point):
                inst = []
                for t in range(len(Event_list)):
                    if t == 0:
                        if Event_list[t][i][j] > h:
                            Event_list[t][i][j] = 2
                        elif Event_list[t][i][j] < -h:
                            Event_list[t][i][j] = 1
                        else:
                            Event_list[t][i][j] = 0
                        inst.append(
                            {'time_since_start': float(t), 'time_since_last_event': 0.0, 'type_event': Event_list[t][i][j]})
                    else:
                        if Event_list[t][i][j] > h:
                            Event_list[t][i][j] = 2
                        elif Event_list[t][i][j] < -h:
                            Event_list[t][i][j] = 1
                        else:
                            Event_list[t][i][j] = 0
                        inst.append(
                            {'time_since_start': float(t), 'time_since_last_event': 1.0, 'type_event': Event_list[t][i][j]})
                data.append(inst)

        return data, len_Event_list


    print('[Info] Loading 04_all(train,val,test) data...')
    train_data, len_Event_list = load_data(A)

    trainloader = get_dataloader(train_data, hawkes_batch, shuffle=False, drop_last=True)

    return trainloader, num_types, len_Event_list

def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """
    Epoch operation in training phase.
    训练阶段的 epoch 运算。
    """

    model.train()
    lambda_list = []

    total_event_ll = 0  # cumulative event log-likelihood 累积事件对数似然
    total_time_se = 0  # cumulative time prediction squared-error 累积时间预测平方误差
    total_event_rate = 0  # cumulative number of correct prediction 累计正确预测数
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (lambda_Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """        # negative log-likelihood
        event_ll, non_event_ll, type_lambda = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        lambda_list.append(type_lambda)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event 我们不能预测第一个事件
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, lambda_list

def train_hawkes(model, training_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    for epoch_i in range(opt.max_epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, lambda_list = train_epoch(model, training_data, optimizer, pred_loss_func, opt)

        lambda_list = np.array(lambda_list)
        print("len_lambda_list:", len(lambda_list))
        print("lambda_list.shape:", lambda_list.shape)
        lambda_list = lambda_list.reshape(94249, 809)

        lambda_mat = []  # lambda_mat是强度函数矩阵
        for t in range(809):
            L = []  # L是t时刻的列表（含307*307）
            for i in range(94249):
                L.append(lambda_list[i][t])
            L_list = []
            for j in range(0, len(L), 307):
                b = L[j:j + 307]
                L_list.append(b)
            lambda_mat.append(L_list)
        np.save(('./04pre_lambda_V5.npy'), lambda_mat)

        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        scheduler.step()
        lambda_mat = np.array(lambda_mat)

    return lambda_mat




