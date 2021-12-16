#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import os
import datetime
import json
import numpy as np
from numpy.linalg import norm
import math
import argparse
from platt import *
from sklearn.metrics import f1_score
import time
import scipy.stats
from itertools import combinations
from tqdm import tqdm
from data_gen import *


class mix_curv_perceptron:

    def __init__(self, mix_component, embed_data, multiclass, max_round, max_update):
        self.X_train = embed_data['X_train']
        self.X_test = embed_data['X_test']
        self.y_train = embed_data['y_train']
        self.y_test = embed_data['y_test']
        self.max_norm = embed_data['max_norm']
        self.curv_value = embed_data['curv_value']

        self.multiclass = multiclass
        self.round = max_round
        self.max_update = max_update
        self.class_labels = list(np.unique(self.y_train))
        self.n_class = len(self.class_labels)
        self.n_train_samples = self.y_train.size
        self.n_test_samples = self.y_test.size
        
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

        self.IpTrain = {}

    def mix_classifier_train(self, idx, error_record, y_bin_train):
        res = 0
        for err_idx in error_record:
            if (err_idx, idx) not in self.IpTrain:
                cur_dis = 0
                start_dim = 0
                for comp_idx in range(len(self.space_type)):
                    if self.space_type[comp_idx] == 'e':
                        cur_dis += np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]], 
                                          self.X_train[idx, start_dim: start_dim + self.space_dim[comp_idx]]) + 1
                    elif self.space_type[comp_idx] == 'h':
                        dist_h = np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]],
                                        self.X_train[idx, start_dim: start_dim + self.space_dim[comp_idx]]) / (self.max_norm[comp_idx] ** 2)
                        if abs(dist_h) > 1:
                            dist_h = np.sign(dist_h)
                        cur_dis += math.sqrt(self.curv_value[comp_idx]) * np.arcsin(dist_h)
                    elif self.space_type[comp_idx] == 's':
                        dist_s = np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]],
                                        self.X_train[idx, start_dim: start_dim + self.space_dim[comp_idx]]) * self.curv_value[comp_idx]
                        if abs(dist_s) > 1:
                            dist_s = np.sign(dist_s)
                        cur_dis += math.sqrt(self.curv_value[comp_idx]) * np.arcsin(dist_s)
                    start_dim += self.space_dim[comp_idx]
                # store the results
                self.IpTrain[(err_idx, idx)] = y_bin_train[err_idx] * cur_dis
            res += error_record[err_idx] * self.IpTrain[(err_idx, idx)]
        return res

    def mix_classifier_test(self, idx, error_record, y_bin_train):
        res = 0
        for err_idx in error_record:
            cur_dis = 0
            start_dim = 0
            for comp_idx in range(len(self.space_type)):
                if self.space_type[comp_idx] == 'e':
                    cur_dis += np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]], 
                                        self.X_test[idx, start_dim: start_dim + self.space_dim[comp_idx]]) + 1
                elif self.space_type[comp_idx] == 'h':
                    dist_h = np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]],
                                    self.X_test[idx, start_dim: start_dim + self.space_dim[comp_idx]]) / (self.max_norm[comp_idx] ** 2)
                    if abs(dist_h) > 1:
                        dist_h = np.sign(dist_h)
                    cur_dis += math.sqrt(self.curv_value[comp_idx]) * np.arcsin(dist_h)
                elif self.space_type[comp_idx] == 's':
                    dist_s = np.dot(self.X_train[err_idx, start_dim: start_dim + self.space_dim[comp_idx]],
                                    self.X_test[idx, start_dim: start_dim + self.space_dim[comp_idx]]) * self.curv_value[comp_idx]
                    if abs(dist_s) > 1:
                        dist_s = np.sign(dist_s)
                    cur_dis += math.sqrt(self.curv_value[comp_idx]) * np.arcsin(dist_s)
                start_dim += self.space_dim[comp_idx]
            res += error_record[err_idx] * y_bin_train[err_idx] * cur_dis
        return res

    def process_data(self):
        if self.multiclass:
            test_probability = np.zeros((self.n_test_samples, self.n_class), dtype=float)
            for class_val in self.class_labels:
                y_bin_train = np.array([1 if val == class_val else -1 for val in self.y_train])
                # initialize the error count dictionary
                tmp_error_record = {0: 1}
                total_error_count = 1
                break_flag = False
                # training
                for epoch in range(self.round):
                    for idx in range(self.n_train_samples):
                        yn = self.mix_classifier_train(idx, tmp_error_record, y_bin_train)
                        if y_bin_train[idx] * yn <= 0:
                            if idx in tmp_error_record:
                                tmp_error_record[idx] += 1
                            else:
                                tmp_error_record[idx] = 1
                            total_error_count += 1
                        print('\r', idx+1, 'samples finished.', total_error_count, end='')
                        if total_error_count == self.max_update:
                            break_flag = True
                            break
                    print('\n', epoch + 1, 'rounds finished.')
                    if break_flag:
                        break
                # obtain the decision values for training samples
                decision_vals = [0] * self.n_train_samples
                for idx in range(self.n_train_samples):
                    decision_vals[idx] = self.mix_classifier_train(idx, tmp_error_record, y_bin_train)
                tmp_ab = SigmoidTrain(deci=decision_vals, label=y_bin_train, prior1=None, prior0=None)
                print('Platt probability computed')
                # testing
                for idx in range(self.n_test_samples):
                    yn = self.mix_classifier_test(idx, tmp_error_record, y_bin_train)
                    test_probability[idx, self.class_labels.index(class_val)] = SigmoidPredict(deci=yn, AB=tmp_ab)
            y_pred_idx = np.argmax(test_probability, axis=1)
            y_pred = np.array([self.class_labels[i] for i in y_pred_idx])
            print('F1 score:', f1_score(self.y_test, y_pred, average='macro'), 'total number of testing samples:', self.y_test.size)
            return f1_score(self.y_test, y_pred, average='macro')
        else:
            error_record = {0: 1}
            total_error_count = 1
            break_flag = False
            # training
            for epoch in range(self.round):
                for idx in tqdm(range(self.y_train.size)):
                    yn = self.mix_classifier_train(idx, error_record, self.y_train)
                    if self.y_train[idx] * yn <= 0:
                        if idx in error_record:
                            error_record[idx] += 1
                        else:
                            error_record[idx] = 1
                        total_error_count += 1
                    # print('\r', f'{idx + 1}/{self.yTrain.size} samples finished.', total_error_count, end='')
                    if total_error_count == self.max_update:
                        break_flag = True
                        break
                print('\n', epoch + 1, 'rounds finished,', total_error_count)
                if break_flag:
                    break
            # testing
            y_pred = []
            for idx in tqdm(range(self.y_test.size)):
                yn = self.mix_classifier_test(idx, error_record, self.y_train)
                if yn > 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            y_pred = np.array(y_pred)
            print('F1 score:', f1_score(self.y_test, y_pred, average='macro'), 'total number of testing samples:', self.y_test.size)
            return f1_score(self.y_test, y_pred, average='macro')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perceptron algorithm in product space form.")
    parser.add_argument("--data_path1", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path2", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path3", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path4", type=str, default=None, help="Where data is located.")
    parser.add_argument("--data_path_num", type=int, default=1, help="How many data path to include.")
    parser.add_argument("--data_name", type=str, default="Lymphoma", help="Which dataset to test on.")
    parser.add_argument("--prod_space", type=str, default="e2,h2,s2", help="Product space form.")
    parser.add_argument("--test_size", type=float, default=0.4, help="Percent of test set size.")
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
        # np.random.seed(0)
        # rnd_idx = list(np.random.permutation(45)[0:10])
        # tmp_labels_chosen_lst = [labels_chosen_lst[i] for i in rnd_idx]
        # labels_chosen_lst = tmp_labels_chosen_lst.copy()
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

    for i in range(label_trails):
        for j in range(args.trails):
            embed_data = mix_data_generation(data_path, prod_space, 2, list(labels_chosen_lst[i]), test_size=args.test_size, cifar_flag=cifar_flag, seed=None, transform=args.transform)
            mix_perp = mix_curv_perceptron(args.prod_space, embed_data, False, 1, 10000)
            print(f'=========={i},{j},{args.prod_space}==========')
            acc[i, j] = mix_perp.process_data()

    print(mean_confidence_interval(acc))
    print('Time used:', time.time() - start)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    cur_time = datetime.datetime.utcnow().isoformat()
    np.save(f'{args.save_path}/{args.data_name}_{prod_space}_perceptron_f1_scores_{cur_time}.npy', acc)

