import torch
import torch.nn as nn

import numpy as np

def adj_loss(pre_adj, lambda_list, A, optimizer, opt):

    pre_adj.train()

    loss_function = torch.nn.MSELoss()

    A_truth = A[1:]
    loss_list = []
    acc_list = []
    arr_mean_list = []

    adj_list = []

    for i in range(len(lambda_list)):

        batch = A_truth[i]
        lambda_mat = lambda_list[i]

        optimizer.zero_grad()

        A_pre = pre_adj(batch, lambda_mat, opt)
        A_pre[A_pre <= 1e-11] = 0
        # A_pre[np.isnan(A_pre)] = 0
        # A_pre[np.isinf(A_pre)] = 0

        if i != len(lambda_list)-1:

            A_tru = torch.FloatTensor(A_truth[i + 1]).to(opt.device)

            loss = loss_function(A_pre, A_tru)
            loss_list.append(loss.item())

            arr = torch.sub(A_pre, A_tru)
            arr_mean = torch.sub(A_tru.mean(0), A_tru)

            acc = torch.sum(abs(arr) < 0.0001) / 28900
            acc_list.append(acc)
            arr_mean_acc = torch.sum(abs(arr_mean) < 0.0001) / 28900
            arr_mean_list.append(arr_mean_acc)

            loss.backward()
            optimizer.step()

        A_pre = A_pre.cpu().detach().numpy()
        adj_list.append(A_pre)

    # l = sum(loss_list) / len(loss_list)
    # print("hawkes_loss:", l)
    # accuracy = sum(acc_list) / len(acc_list)
    # print("hawkes_accuracy:", accuracy.item())
    # mean_accuracy = sum(arr_mean_list) / len(arr_mean_list)
    # print("mean_accuracy:", mean_accuracy.item())

    adj_list.insert(0, A[0])
    adj_list = np.array(adj_list)
    #print("adj_list.shape:", adj_list.shape)

    #np.save('./08_all_pre_adj_V1.npy', adj_list)
    A_pre = adj_list

    #return l, accuracy
    return A_pre

def train_adj(pre_adj, lambda_list, A, optimizer, scheduler, opt):
    """ Start training. """

    for epoch_i in range(opt.max_epoch):
        epoch = epoch_i + 1
        #print('[ Epoch', epoch, ']')

        # start = time.time()

        #loss, accuracy = adj_loss(pre_adj, lambda_list, A, optimizer, opt)
        A_pre = adj_loss(pre_adj, lambda_list, A, optimizer, opt)
        # loss = sum(loss_list)/len(loss_list)

        # print('  - (Training)    aaccuracy: {type: 8.5f}, '
        #       'loss: {type: 8.5f}, '
        #       'elapse: {elapse:3.3f} min'
        #       .format(type=accuracy, l=loss, elapse=(time.time() - start) / 60))

        scheduler.step()

    return A_pre

class Pre_adj(nn.Module):
    """
    生成邻接矩阵
    输入：training_data, all_lambda
    输出：邻接矩阵
    """
    def __init__(self, num_point):
        super(Pre_adj, self).__init__()
        self.input_size = num_point
        self.O = nn.Parameter(torch.FloatTensor(self.input_size, self.input_size), requires_grad=True)

        torch.nn.init.xavier_normal_(self.O, gain=1)

    def forward(self, A, lambda_mat, opt):

        #pre_adj = []
        # print("len(lambda_mat):", len(lambda_mat))

        #data = torch.FloatTensor([item.cpu().detach().numpy() for item in lambda_mat]).cuda()

        # for k in range(len(lambda_mat)):
        #     data = torch.FloatTensor(lambda_mat[k]).to(opt.device)
        #     pre = self.O.mul(data) + A[k]
        #     pre_adj.append(pre)
        # adj = torch.FloatTensor([item.cpu().detach().numpy() for item in pre_adj]).cuda()

        batch = torch.FloatTensor(A).to(opt.device)
        # batch.requires_grad_(True)
        lambda_nft = torch.FloatTensor(lambda_mat).to(opt.device)
        # lambda_list.requires_grad_(True)

        pre = self.O.mul(lambda_nft) + batch
        return pre