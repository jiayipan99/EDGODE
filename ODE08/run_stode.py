import sys
sys.path.append('/home/user/pan/project')

import os
import random
import numpy as np
from scipy.fft import set_global_backend
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import time
from tqdm import tqdm
import dgl
from loguru import logger

from scipy.sparse import csr_matrix

from args import args
from model import ODEGCN_C
from utils import generate_dataset, read_data, get_normalized_adj, all_data, generate_all_data
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

from metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def train(loader, model, optimizer, criterion, device, sp_g, A_sp_wave):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device) #(16,3,170,24)
        targets = targets.to(device) #(16,170,12)
        g = sp_g[idx]
        A_hat = A_sp_wave[idx]

        inputs = inputs.permute(0, 2, 3, 1)
        outputs = model(inputs, g, A_hat)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model,device, sp_g, A_sp_wave):
    batch_rmse_loss = 0
    batch_mae_loss = 0
    batch_mape_loss = 0

    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        g = sp_g[idx]
        A_hat = A_sp_wave[idx]

        inputs = inputs.permute(0, 2, 3, 1)

        output = model(inputs, g, A_hat)

        out_unnorm = output.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)

        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)

def main(args):
    # random seed
    seed = 2000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cuda:' + str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    all_data_load, true_value, train_loader, valid_loader, test_loader = generate_all_data(args)

    A_lst = np.load('08pre_adj_V8_1.npy')
    # A[np.isnan(A)] = 0.
    # A[np.isinf(A)] = 0.

    data, dtw_matrix, sp_matrix = read_data(args, A_lst)

    sp_g = []
    for i in range(len(sp_matrix)):
        mat = csr_matrix(sp_matrix[i])
        sp_g_mat = dgl.from_scipy(mat).to(device)
        sp_g.append(sp_g_mat)

    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)

    A_sp_wave = []
    for j in range(len(sp_matrix)):
        A_sp = get_normalized_adj(sp_matrix[j]).to(device)
        A_sp_wave.append(A_sp)

    net = ODEGCN_C(num_nodes=data.shape[1],
                  num_features=data.shape[2],
                  num_timesteps_input=args.his_length,
                  num_timesteps_output=args.pred_length
                  )

    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_valid_rmse = 10000
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    g1_train = sp_g[0:518]
    A_hat1_train = A_sp_wave[0:518]

    g1_valid = sp_g[518:691]
    A_hat1_valid = A_sp_wave[518:691]

    g1_test = sp_g[691:864]
    A_hat1_test = A_sp_wave[691:864]

    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        loss = train(train_loader, net, optimizer, criterion, device, g1_train, A_hat1_train)
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, device, g1_train, A_hat1_train)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, device, g1_valid, A_hat1_valid)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

        if args.log:
            logger.info(f'\n##on train data## loss: {loss}, \n' + 
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        
        scheduler.step()

    net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
    test_rmse, test_mae, test_mape = eval(test_loader, net, device, g1_test, A_hat1_test)
    if args.log:
        logger.info(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')
    else:
        print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')

if __name__ == '__main__':
    main(args)

