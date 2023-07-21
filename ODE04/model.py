import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import SAGEConv

from ODE04_notcn.odegcn import ODEG


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it 额外的尺寸将增加填充，删除它
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(GCN, self).__init__()
        #self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        y = torch.einsum('ij, kjlm-> kilm', A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            时间块中的特征通道列表，最后一个是输出特征通道
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        # self.temporal1 = TemporalConvNet(num_inputs=in_channels,
        #                                  num_channels=out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels[-1])
        )
        self.odeg = ODEG(out_channels[-1], 24, time=6)

        self.batch_norm = nn.BatchNorm2d(num_nodes)


    def forward(self, X, A_hat):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        x_shape = X.shape
        X = X.reshape(-1, x_shape[3])  # Reshape input to (batch_size*num_nodes*num_timesteps, num_features)
        t = self.mlp(X)
        t = t.view(x_shape[0], x_shape[1], x_shape[2],
                   -1)  # Reshape back to (batch_size, num_nodes, num_timesteps, out_channels[-1])
        t = self.odeg(t, A_hat)
        t = F.relu(t)

        return self.batch_norm(t)


class SAGE(nn.Module):
    """ the overall network framework """

    def __init__(self, in_features, out_features):
        super(SAGE, self).__init__()
        self.sage = SAGEConv(in_features, out_features, 'pool')
        # self.g = g

    def forward(self, x, g):
        x = x.permute(1, 0, 2, 3)
        x = self.sage(g, x)
        x = x.permute(1, 0, 2, 3)

        return x


class ODEGCN_C(nn.Module):
    """ the overall network framework """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        Args:
            num_nodes : number of nodes in the graph图中节点的数量
            num_features : number of features at each node in each time step每个时间步中每个节点上的特征数量
            num_timesteps_input : number of past time steps fed into the network输入到网络的过去时间步数
            num_timesteps_output : desired number of future time steps output by the network网络输出的期望未来时间步数
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """

        super(ODEGCN_C, self).__init__()

        self.sp_blocks1 = nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                           num_nodes=num_nodes),
                STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                           num_nodes=num_nodes)
            )

        self.sp_blocks2 = nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                       num_nodes=num_nodes),
            STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                       num_nodes=num_nodes)
        )

        self.sp_blocks3 = nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                       num_nodes=num_nodes),
            STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                       num_nodes=num_nodes)
        )

        self.sp_blocks4 = nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                       num_nodes=num_nodes),
            STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                       num_nodes=num_nodes)
        )

        self.sp_blocks5 = nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                       num_nodes=num_nodes),
            STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                       num_nodes=num_nodes)
        )

        self.sp_blocks6 = nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 48, 64],
                       num_nodes=num_nodes),
            STGCNBlock(in_channels=64, out_channels=[64, 48, 64],
                       num_nodes=num_nodes)
        )

        self. sage1 = SAGE(in_features=64, out_features=64)

        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output)
        )

    def forward(self, x, g, A_hat):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        outs = []
        x6, x5, x4, x3, x2 = x, x, x, x, x

        # spatial graph
        for blk in self.sp_blocks1:
            x = blk(x, A_hat)
        b = self.sage1(x, g)
        outs.append(b)

        for blk in self.sp_blocks2:
            x2 = blk(x2, A_hat)
        b = self.sage1(x2, g)
        outs.append(b)

        for blk in self.sp_blocks3:
            x3 = blk(x3, A_hat)
        b = self.sage1(x3, g)
        outs.append(b)

        # for blk in self.sp_blocks4:
        #     x4 = blk(x4, A_hat)
        # b = self.sage1(x4, g)
        # outs.append(b)
        #
        # for blk in self.sp_blocks5:
        #     x5 = blk(x5, A_hat)
        # b = self.sage1(x5, g)
        # outs.append(b)
        #
        # for blk in self.sp_blocks6:
        #     x6 = blk(x6, A_hat)
        # b = self.sage1(x6, g)
        # outs.append(b)

        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x)

        return x