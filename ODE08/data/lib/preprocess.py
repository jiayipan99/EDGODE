# -*- coding:utf-8 -*-

import numpy as np
from ODE.data.lib.utils import get_sample_indices
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import random
def normalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0, 2, 1, 3)
    val = (val).transpose(0, 2, 1, 3)
    test = (test).transpose(0, 2, 1, 3)

    return {'mean': mean, 'std': std}, train, val, test

def read_and_generate_dataset(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (17856, 170, 3)

    all_samples = []
    for idx in tqdm(range(data_seq.shape[0])):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))
    # random.seed(21)

    num_data = len(all_samples)
    indices = list(range(num_data))
    split1 = int(num_data * 0.6)
    split2 = int(num_data * 0.8)
    random.shuffle(indices)
    # print(indices)
    train_indices,val_indices,test_indices = indices[:split1],indices[split1:split2], indices[split2:]
    #print('1111')
    #
    # # 创建 SubsetRandomSampler 对象
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    #
    # train_loader = DataLoader(all_samples, batch_size=16, sampler=train_sampler)
    # val_loader = DataLoader(all_samples, batch_size=16, sampler=val_sampler)
    # test_loader = DataLoader(all_samples, batch_size=16, sampler=test_sampler)

    # split_line1 = int(len(all_samples) * 0.6)
    # split_line2 = int(len(all_samples) * 0.8)
    if not merge:
        td = []
        for index in train_indices:
            td.append(all_samples[index])
        training_set = [np.concatenate(i, axis=0)for i in zip(*td)]
    # else:
    #     for index in train_indices:
    #
    #         print('Merge training set and validation set!')
    #         training_set = [np.concatenate(i, axis=0)
    #                         for i in zip(*all_samples[index])]
    vd =[]
    for index in val_indices:
        vd.append(all_samples[index])
    validation_set = [np.concatenate(i, axis=0)for i in zip(*vd)]
    ted=[]
    for index in test_indices:
        ted.append(all_samples[index])
    testing_set = [np.concatenate(i, axis=0) for i in zip(*ted)]

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    print("                  hour_shape         target_shape")
    print('train data   {}   {}'.format(train_hour.shape, train_target.shape))
    print('valid data   {}   {}'.format(val_hour.shape, val_target.shape))
    print('tests data   {}   {}'.format(test_hour.shape, test_target.shape))

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data