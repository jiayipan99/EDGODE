import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import torch
from evolution_KL_ST import setup_features_tuple,setup_Adj_matrix
from data.lib.preprocess import read_and_generate_dataset


files = {
    'pems03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'pems04': ['PEMS04/pems04.npz', 'PEMS04/distance.csv'],
    'pems07': ['PEMS07/pems07.npz', 'PEMS07/distance.csv'],
    'pems08': ['PEMS08/pems08.npz', 'PEMS08/distance.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}

def all_data(args):
    filename = args.filename
    file = files[filename]  # FILE:['PEMS08/pems08.npz','PEMS08/distance.csv']
    filepath = r"./data/"
    if args.remote:
        filepath = r'/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + file[0])['data']  # (16992,307,3)
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)

    return torch.from_numpy(data.astype(np.float32))


def read_data(args, A_lst):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    file = files[filename] #FILE:['PEMS08/pems08.npz','PEMS08/distance.csv']
    filepath = r"./data/"
    if args.remote:
        filepath = r'/home/lantu.lqq/ftemp/data/'
    data = np.load(filepath + file[0])['data']#(16992,307,3)

    sp_matrix = A_lst
    sp_matrix[np.isnan(sp_matrix)] = 0.
    sp_matrix[np.isinf(sp_matrix)] = 0.
    dtw_matrix = sp_matrix

    return data, dtw_matrix, sp_matrix


def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    返回一个张量，度归一化邻接矩阵。
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


class MyDataset(Dataset):
    def __init__(self, data, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[split_start: split_end]
        self.his_length = his_length
        self.pred_length = pred_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length].permute(1, 0, 2) #(307,12,3)
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0].permute(1, 0) #(307,12)
        return torch.Tensor(x), torch.Tensor(y)
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def generate_dataset(data, args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length
    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

def generate_all_data(args):
    graph_signal_matrix_filename = 'data/PEMS0%s/pems0%s.npz' % (args.data_name, args.data_name)
    points_per_hour = 12
    num_for_predict = 12
    num_of_weeks = 2
    num_of_days = 1
    num_of_hours = 2
    merge = False

    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = all_data['test']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=args.batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=args.batch_size,
        shuffle=False
    )
    return all_data, true_value, train_loader, val_loader, test_loader
