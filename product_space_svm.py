#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import argparse
import datetime
import cvxpy as cp
import numpy as np
import math
from sklearn.metrics import f1_score
from numpy.linalg import norm, eigh
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import json
from itertools import combinations
from data_gen import *
import warnings
warnings.filterwarnings("ignore")

# define a small constant for normalization
eps = 1e-6


class mix_curv_svm:

    def __init__(self, mix_component, embed_data):
        self.X_train = embed_data['X_train']
        self.X_test = embed_data['X_test']
        self.y_train = embed_data['y_train']
        self.y_test = embed_data['y_test']
        self.curv_value = embed_data['curv_value']
        self.train_size = self.y_train.size
        self.test_size = self.y_test.size
        # other parameters
        self.alpha_e = 1
        self.alpha_s = 1
        self.alpha_h = 1
        self.r = 0.01
        # store each component in order
        prod_space_component = mix_component.split(',')
        self.space_type = []
        self.space_dim = []
        for comp in prod_space_component:
            self.space_type.append(comp[0])
            if comp.startswith('e'):
                self.space_dim.append(int(comp[1]))
            else:
                self.space_dim.append(int(comp[1]) + 1)
        # Construct train and test matrices
        self.G_train = np.zeros((self.train_size, self.train_size))
        self.G_train_list = []
        self.G_test = np.zeros((self.train_size, self.test_size))
        self.G_test_list = []
        start_dim = 0
        for comp_idx in range(len(self.space_type)):
            train_matrix = self.X_train[:, start_dim: start_dim + self.space_dim[comp_idx]]
            test_matrix = self.X_test[:, start_dim: start_dim + self.space_dim[comp_idx]]
            if self.space_type[comp_idx] == 'e':
                Ge_train = np.matmul(train_matrix, train_matrix.T)
                self.G_train += Ge_train
                self.G_train_list.append(Ge_train)
                Ge_test = np.matmul(train_matrix, test_matrix.T)
                self.G_test += Ge_test
                self.G_test_list.append(Ge_test)
            elif self.space_type[comp_idx] == 'h':
                R = max(np.sqrt(np.max(np.matmul(train_matrix, train_matrix.T))),
                        np.sqrt(np.max(np.matmul(train_matrix, test_matrix.T)))) + eps
                Gh_train = np.matmul(train_matrix, train_matrix.T)
                Gh_train = np.arcsin(Gh_train / (R ** 2))
                self.G_train += Gh_train
                self.G_train_list.append(Gh_train)
                Gh_test = np.matmul(train_matrix, test_matrix.T)
                Gh_test = np.arcsin(Gh_test / (R ** 2))
                self.G_test += Gh_test
                self.G_test_list.append(Gh_test)
            elif self.space_type[comp_idx] == 's':
                Cs = self.curv_value[comp_idx]
                Gs_train = np.matmul(train_matrix, train_matrix.T)
                Gs_train = np.arcsin((Cs * Gs_train) / (abs(Cs * Gs_train).max() + eps))
                self.G_train += Gs_train
                self.G_train_list.append(Gs_train)
                Gs_test = np.matmul(train_matrix, test_matrix.T)
                Gs_test = np.arcsin((Cs * Gs_test) / (abs(Cs * Gs_test).max() + eps))
                self.G_test += Gs_test
                self.G_test_list.append(Gs_test)
            start_dim += self.space_dim[comp_idx]
    
    def process_data(self, solver_type='SCS'):
        Y = np.diagflat(self.y_train)
        zeta = cp.Variable(self.train_size)
        beta = cp.Variable(self.train_size)
        epsilon = cp.Variable(1)

        conds = [epsilon >= 0, zeta >= 0, Y @ (self.G_train @ beta + cp.sum(beta)) >= epsilon - zeta]
        for comp_idx in range(len(self.space_type)):
            if self.space_type[comp_idx] == 'e':
                conds.append(cp.quad_form(beta, self.G_train_list[comp_idx]) <= self.alpha_e ** 2)
            elif self.space_type[comp_idx] == 's':
                conds.append(cp.quad_form(beta, self.G_train_list[comp_idx]) <= math.pi / 2)

        prob = cp.Problem(cp.Minimize(-epsilon + cp.sum(zeta)), conds)
        prob.solve(solver=solver_type)

        beta = beta.value
        epsilon = epsilon.value
        zeta = zeta.value
        # assume all alpha's are 1
        y_pred = np.sign(np.matmul(beta.T, self.G_test) + np.sum(beta))
        y_pred = y_pred.reshape((self.test_size,)).astype(int)
        score = f1_score(self.y_test, y_pred, average='macro')
        print(f'Mix curv SVM F1 score: {score}')
        return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SVM algorithm in product space form.")
    parser.add_argument("--data_path1", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path2", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path3", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path4", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path_num", type=int, default=1, help="How many data path to include.")
    parser.add_argument("--data_name", type=str, default="Lymphoma", help="Which dataset to test on.")
    parser.add_argument("--prod_space", type=str, default="e2,h2,s2", help="Product space form.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Percent of test set size.")
    parser.add_argument("--trails", type=int, default=10, help="Number of trails want to repeat.")
    parser.add_argument("--save_path", type=str, default="results", help="Where to save results.")
    parser.add_argument("--transform", type=bool, default=False, help="Where to perform inverse projection.")
    args = parser.parse_args()

    start = time.time()
    cifar_flag = False
    if args.data_name == "Lymphoma":
        labels_chosen_lst = [[0, 1]]
    elif args.data_name == "Blood_cell_landmark":
        labels_chosen_lst = list(combinations([i for i in range(10)], 2))
        # for debug only
        # rnd_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        # labels_chosen_lst = [labels_chosen_lst[i] for i in rnd_idx]
    elif args.data_name == "cifar100":
        cifar_flag = True
        labels_chosen_lst = []
        for i in range(30):
            np.random.seed(i)
            labels_chosen_lst.append(list(np.random.permutation(100)[0:2]))
    else:
        # used for debugging purpose
        labels_chosen_lst = [[0, 1]]

    label_trails = len(labels_chosen_lst)
    acc = np.zeros((label_trails, args.trails))
    # path to different files
    data_path = [args.data_path1, args.data_path2, args.data_path3, args.data_path4]
    data_path = data_path[0: args.data_path_num]
    print(data_path)
    # curvature of each file
    prod_space = []
    for file_name in data_path:
        if cifar_flag:
            prod_space.append(file_name.split('-')[2])
        else:
            prod_space.append(file_name.split('-')[3])
    joint_prod_space = ','.join(prod_space)
    assert args.prod_space == joint_prod_space

    valid_acc = []
    valid_trails = []
    invalid_trails = []

    for i in range(label_trails):
        for j in range(args.trails):
            embed_data = mix_data_generation(data_path, prod_space, 2, list(labels_chosen_lst[i]), svm_flag=True, cifar_flag=cifar_flag, seed=j, transform=args.transform)
            mix_svm = mix_curv_svm(args.prod_space, embed_data)
            # print(f'=========={i},{j}==========')
            try:
                acc[i, j] = mix_svm.process_data(solver_type='ECOS')
                valid_acc.append(acc[i, j])
                valid_trails.append((i, j))
            except:
                try:
                    acc[i, j] = mix_svm.process_data(solver_type='SCS')
                    valid_acc.append(acc[i, j])
                    valid_trails.append((i, j))
                except:
                    invalid_trails.append((i, j))

    print(f'=========={args.prod_space}==========')
    print(f'Valid trails number:', len(valid_acc))
    print(mean_confidence_interval(np.array(valid_acc)))
    print('Time used:', time.time() - start)
    print('======================================')
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    cur_time = datetime.datetime.utcnow().isoformat()
    np.savez(f'{args.save_path}/{args.data_name}_{prod_space}_svm_f1_scores_{cur_time}.npz', acc=acc, valid_trails=valid_trails)
