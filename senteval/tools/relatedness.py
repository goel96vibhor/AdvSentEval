# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Semantic Relatedness (supervised) with Pytorch
"""
from __future__ import absolute_import, division, unicode_literals

import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim

from scipy.stats import pearsonr
import pickle

class RelatednessPytorch(object):
    # Can be used for SICK-Relatedness, and STS14
    def __init__(self, train, valid, test, devscores, config):
        # fix seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        torch.cuda.manual_seed(config['seed'])

        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        self.inputdim = train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.l2reg = 0.
        self.batch_size = 8
        self.maxepoch = 1000
        self.early_stop = True
        self.model_name = config['model_name'] if config['model_name'] is not None else ""
        self.task_name = config['task_name'] if config['task_name'] is not None else ""
        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
            nn.Softmax(dim=-1),
        )
        self.loss_fn = nn.MSELoss()
        self.model_loaded = False


        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)

    def prepare_data(self, trainX, trainy, devX, devy, testX, testy):
        # Transform probs to log-probs for KL-divergence
        trainX = torch.from_numpy(trainX).float().cuda()
        trainy = torch.from_numpy(trainy).float().cuda()
        devX = torch.from_numpy(devX).float().cuda()
        devy = torch.from_numpy(devy).float().cuda()
        testX = torch.from_numpy(testX).float().cuda()
        testY = torch.from_numpy(testy).float().cuda()

        return trainX, trainy, devX, devy, testX, testy

    def run(self):
        self.nepoch = 0
        bestpr = -1
        early_stop_count = 0
        r = np.arange(1, 6)
        stop_train = False
        filename = 'models/finalized_model_' + self.model_name + '_' + self.task_name + '_.sav'

        # Preparing data
        trainX, trainy, devX, devy, testX, testy = self.prepare_data(
            self.train['X'], self.train['y'],
            self.valid['X'], self.valid['y'],
            self.test['X'], self.test['y'])

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(trainX, trainy, nepoches=50)
            yhat = np.dot(self.predict_proba(devX), r)
            pr = pearsonr(yhat, self.devscores)[0]
            pr = 0 if pr != pr else pr  # if NaN bc std=0
            # early stop on Pearson
            if pr > bestpr:
                bestpr = pr
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel

        pickle.dump(self.model, open(filename, 'wb'))
        test_preds = self.predict_proba(testX)
        yhat = np.dot(test_preds, r)
        print(test_preds.shape)

        # self.model = pickle.load(open(filename, 'rb'))
        # test_preds =[]
        # test_preds.extend(self.predict_batch_proba(self.test['X']))
        # test_preds = np.array(test_preds)
        # yhat = np.dot(test_preds, r)
        # print(test_preds.shape)

        return bestpr, yhat

    def predict(self, test_x):

        r = np.arange(1, 6)
        filename = 'models/finalized_model_' + self.model_name + '_' + self.task_name + '_.sav'


        if self.model_loaded == False :
            self.model = pickle.load(open(filename, 'rb'))
            self.model_loaded = True
        test_preds =[]
        test_preds.extend(self.predict_batch_proba(test_x))
        test_preds = np.array(test_preds)
        yhat = np.dot(test_preds, r)
        print(test_preds.shape)

        return yhat


    def trainepoch(self, X, y, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
                Xbatch = X[idx]
                ybatch = y[idx]
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                if len(probas) == 0:
                    probas = self.model(Xbatch).data.cpu().numpy()
                else:
                    probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
        return probas


    def predict_batch_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                Xbatch = torch.from_numpy(Xbatch).float().cuda()
                if len(probas) == 0:
                    probas = self.model(Xbatch).data.cpu().numpy()
                else:
                    probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
        return probas