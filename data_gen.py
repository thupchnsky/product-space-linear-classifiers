#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import os
import json
import numpy as np
from numpy.linalg import norm
import math
import torch
from sklearn.model_selection import train_test_split
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = data.flatten()
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def minkovski_ip(x, y):
    assert x.shape == y.shape
    res = -x[0]*y[0]
    for i in range(1, x.size):
        res += x[i] * y[i]
    return res


def stereo_project(x, c):
    return x[1:] / (1 + math.sqrt(abs(c)) * x[0])


def stereo_inv_project(y, c):
    x = np.zeros(y.size + 1)
    x[0] = (1 - c * norm(y)**2) / (math.sqrt(abs(c)) * (1 + c * norm(y)**2))
    x[1:] = 2 * y / (1 + c * norm(y)**2)
    return x


def mix_data_generation(mix_data_path_list, mix_component_list, num_class, labels_chosen, test_size=0.2, svm_flag=False, cifar_flag=False, seed=None, transform=False):
    curv_value = []
    component_data_list = []
    skip_idx = []
    max_norm = []
    for file_idx in range(len(mix_data_path_list)):
        mix_data_path = mix_data_path_list[file_idx]
        mix_component = mix_component_list[file_idx]
        prod_space_component = mix_component.split(',')
        print(prod_space_component)
        with open(os.path.join(mix_data_path, 'stat_dict.json'), 'r') as fp:
            curv_data = json.load(fp)

        for i in range(len(prod_space_component)):
            # test data (whole dataset for sc)
            component_data_path_test = os.path.join(mix_data_path, 'repr', f'test_comp_00{i}_{prod_space_component[i]}.pt')
            component_data_test = torch.load(component_data_path_test)
            component_data_test = component_data_test.detach().numpy()
            if transform and not prod_space_component[i].startswith('e'):
                tmp_component = np.zeros((component_data_test.shape[0], component_data_test.shape[1]+1))
                for sample_idx in range(tmp_component.shape[0]):
                    tmp_component[sample_idx, :] = stereo_inv_project(component_data_test[sample_idx, :], curv_data[f'comp_00{i}_{prod_space_component[i]}/curvature'])
                component_data_test = tmp_component.copy()
            if cifar_flag:
                # train data
                component_data_path_train = os.path.join(mix_data_path, 'repr', f'train_comp_00{i}_{prod_space_component[i]}.pt')
                component_data_train = torch.load(component_data_path_train)
                component_data_train = component_data_train.detach().numpy()
                if transform and not prod_space_component[i].startswith('e'):
                    tmp_component = np.zeros((component_data_train.shape[0], component_data_train.shape[1]+1))
                    for sample_idx in range(tmp_component.shape[0]):
                        tmp_component[sample_idx, :] = stereo_inv_project(component_data_train[sample_idx, :], curv_data[f'comp_00{i}_{prod_space_component[i]}/curvature'])
                    component_data_train = tmp_component.copy()
                component_data = np.concatenate([component_data_train, component_data_test], axis=0).copy()
            else:
                component_data = component_data_test.copy()
            component_data_list.append(component_data)
            curv_value.append(abs(curv_data[f'comp_00{i}_{prod_space_component[i]}/curvature']))
            max_norm.append(norm(component_data, axis=1).max())
            # check if embedding of each point is valid
            if prod_space_component[i].startswith('h'):
                for j in range(component_data.shape[0]):
                    if j not in skip_idx:
                        fg = abs(minkovski_ip(component_data[j], component_data[j]) + 1 / curv_value[-1]) < 1e-2
                        if not fg:
                            skip_idx.append(j)
            elif prod_space_component[i].startswith('s'):
                for j in range(component_data.shape[0]):
                    if j not in skip_idx:
                        fg = abs(norm(component_data[j]) ** 2 - 1 / curv_value[-1]) < 1e-2
                        if not fg:
                            skip_idx.append(j)

    # concatenate each component
    X = np.concatenate(component_data_list, axis=1)
    # load the labels
    labels_test = torch.load(os.path.join(mix_data_path_list[0], 'repr', 'test_labels.pt'))
    labels_test = labels_test.detach().numpy()
    if cifar_flag:
        labels_train = torch.load(os.path.join(mix_data_path_list[0], 'repr', 'train_labels.pt'))
        labels_train = labels_train.detach().numpy()
        labels = np.concatenate([labels_train, labels_test]).copy()
    else:
        labels = labels_test.copy()
    
    active_idx = []
    for i in range(labels.size):
        if i not in skip_idx and labels[i] in labels_chosen:
            active_idx.append(i)
    print(f'Original data size: {X.shape[0]}, valid data size: {len(active_idx)}')
    X = X[active_idx, :]
    labels = labels[active_idx]

    if svm_flag:
        test_size = 1 - 150 / labels.size

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=seed, stratify=labels)

    # change labels to 1 and -1 for binary classification problem
    pos_label = y_train[0]
    if num_class == 2:
        y_train = np.array([1 if val == pos_label else -1 for val in y_train])
        y_test = np.array([1 if val == pos_label else -1 for val in y_test])

    embed_data = {}
    embed_data['X_train'] = X_train
    embed_data['X_test'] = X_test
    embed_data['max_norm'] = max_norm
    embed_data['curv_value'] = curv_value
    embed_data['y_train'] = y_train
    embed_data['y_test'] = y_test

    return embed_data


if __name__ == '__main__':
    chkpt_path = 'vae-sc-Lymphoma_processed-s2-2021-10-28T21:09:51.028008'
    # embed_data = mix_data_generation('data/cifar100/e2,h2,s2/', 'e2,h2,s2', 2, [0, 1])
