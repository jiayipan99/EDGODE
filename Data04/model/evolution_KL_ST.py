import numpy as np
import time
import itertools
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.stats import entropy


def setup_features_tuple(train_w_d_h):
    """  train_w_d_h: train_week / train_day / train_hour， from last layer  """
    for i, unit in enumerate(train_w_d_h):
        feature_0 = unit[0]
        feature_1 = unit[1]
        feature_2 = unit[2]

        nodes_f012 = {}

        for index, (i, j, k) in enumerate(zip(feature_0, feature_1, feature_2)):
            nodes_f012[index] = [i, j, k]

        for key in nodes_f012.keys():
            temp = []
            for j in range(len(nodes_f012[key][0])):
                x = nodes_f012[key][0][j]
                # y = float(nodes_f012[key][1][j])
                # z = float(nodes_f012[key][2][j])
                # temp.append((np.exp(x), np.exp(y), np.exp(z)))
                # temp.append((x, y, z))
                temp.append((x,))
            nodes_f012[key] = temp
            
    return nodes_f012


def setup_Adj_matrix(node_features012, nodes_count):
    adj_value = np.zeros((nodes_count, nodes_count))

    for key1, key2 in itertools.product(node_features012.keys(), node_features012.keys()):
        if np.abs(np.array(node_features012[key2][0]).mean() - np.array(node_features012[key1][0]).mean()) > 80.:
            adj_value[key1][key2] = 0.
        else:
            if key1 != key2:
                mem_for_t = {}

                # 计算 S 值：KLD
                S = entropy(node_features012[key1], node_features012[key2])

                # 计算 d  值
                dist_matrix = np.load('pems04_spatial_distance.npy')
                std = np.std(dist_matrix[dist_matrix != np.float('inf')])
                mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
                dist_matrix = (dist_matrix - mean) / std
                sigma = 10
                sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
                # sp_matrix[sp_matrix < 0.5] = 0

                d = sp_matrix[key1][key2]

                # adj_value[key1][key2] = S / d if d != 0 else 0.
                adj_value[key1][key2] = S + d

    # for key1, key2 in itertools.product(node_features012.keys(), node_features012.keys()):
    #     if key1 != key2:
    #         mem_for_t = {}
    #
    #         # 计算 S 值：KLD
    #         S = entropy(node_features012[key1], node_features012[key2])
    #         # min_S = np.min(S)
    #         # max_S = np.max(S)
    #         # S = (S - min_S) / (max_S - min_S + 1e-8)
    #
    #         # 计算 d  值
    #         dist_matrix = np.load('pems04_spatial_distance.npy')
    #         std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #         mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #         dist_matrix = (dist_matrix - mean) / std
    #         # min_sp_matrix = np.min(dist_matrix[dist_matrix != np.float('inf')])
    #         # max_sp_matrix = np.max(dist_matrix[dist_matrix != np.float('inf')])
    #         # dist_matrix = (dist_matrix - min_sp_matrix) / (max_sp_matrix - min_sp_matrix)
    #         sigma = 10
    #         sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    #         # sp_matrix[sp_matrix < 0.5] = 0
    #
    #         d = sp_matrix[key1][key2]
    #
    #         # adj_value[key1][key2] = S / d if d != 0 else 0.
    #         adj_value[key1][key2] = S + d

#     if no A_norm:
#         adj_value_0 = normalize(adj_value, axis=0, norm='max')
#         adj_value_1 = normalize(adj_value, axis=1, norm='max')
#         adj_value = (adj_value_1 + adj_value_0) / 24

#     adj_value[adj_value < theta] = gamma

    return adj_value

