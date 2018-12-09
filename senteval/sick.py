# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SICK Relatedness and Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

from senteval.tools.relatedness import RelatednessPytorch
from senteval.tools.validation import SplitClassifier
from torch import nn
import torch
import torch.optim as optim
import copy
import pickle

class SICKRelatednessEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}
        self.task_name = 'SICKrelated'


    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + \
                  self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + \
                  self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data


    def generate_adv_samples(self, sst_embed_x, sst_embed_y):

        adv_embed_x = []
        adv_embed_y = []
        adv_sentences = []
        adv_batch_size = self.params.batch_size

        total_samples = len(sst_embed_x)
        # total_samples = 100
        # adv_batch_size = total_samples
        for stidx in range(0, total_samples, adv_batch_size):

            batch = self.sst_data['test']['X'][stidx:stidx + adv_batch_size]
            batch_labels = self.sst_data['test']['y'][stidx:stidx + adv_batch_size]
            batch_embeds = sst_embed_x[stidx:stidx + adv_batch_size]

            print("Computing adversarial samples for batch: %d no of sentences %d" %(stidx/adv_batch_size, len(batch) ))

            modified_vecs, repeated_labels, adv_batch_sentences = self.adversarialFunc(self.params, batch, batch_labels, batch_embeds)

            for sentence_adversary_embeds, sentence_labels, sentence_adversaries in zip(modified_vecs, repeated_labels, adv_batch_sentences):
                adv_embed_x.append(sentence_adversary_embeds)
                adv_embed_y.append(sentence_labels)
                adv_sentences.append(sentence_adversaries)

            print("%d sentences done"%(stidx))
        print("adv_embed length:%d %d"%(len(adv_embed_x), len(adv_embed_y)))
        return adv_embed_x, adv_embed_y, adv_sentences


    def train_y_pred_model(self, train_x, dev_x, train_y, dev_y):

        trainf = np.c_[np.abs(train_x['advs_x'] - train_x['orig_x']), train_x['advs_x'] * train_x['orig_x'], train_x['y_hat']]

        devf = np.c_[
            np.abs(dev_x['advs_x'] - dev_x['orig_x']), dev_x['advs_x'] * dev_x['orig_x'], dev_x['y_hat']]



        inputDim = trainf.shape[1]
        print("Training y pred model with shape:",inputDim)
        self.adv_model = nn.Sequential(
            nn.Linear(inputDim, 1),
        )
        filename = 'models/y_prediction_model_' + self.params.model_name + '_' + self.task_name + '_.sav'
        self.l2reg = 0.
        self.batch_size = 16
        self.maxepoch = 1000
        self.early_stop = True
        stop_train = False
        self.loss_fn = nn.MSELoss()
        self.nepoch = 0
        if torch.cuda.is_available():
            self.adv_model = self.adv_model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.adv_model.parameters(),
                                    weight_decay=self.l2reg)
        early_stop_count = 0
        best_mse = 6
        mse = mean_squared_error(dev_x['y_hat'], dev_y)
        # early stop on Pearson
        print("Initial mean squared error:", mse)


        while not stop_train and self.nepoch <= self.maxepoch:

            self.trainepoch(trainf, train_y, nepoches=1)
            dev_yhat = self.predict_proba(devf)

            mse = mean_squared_error(dev_yhat, dev_y)
            # early stop on Pearson
            print("mean squared error:", mse)
            if mse < best_mse:
                best_mse = mse
                bestmodel = copy.deepcopy(self.adv_model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1

        self.adv_model = bestmodel
        pickle.dump(self.adv_model, open(filename, 'wb'))




    def trainepoch(self, X, y, nepoches=1):
        self.adv_model.train()
        for i in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs = []
            print("Epoch no:",i)
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
                Xbatch = X[idx]
                Xbatch = torch.from_numpy(Xbatch).float().cuda()

                if i + self.batch_size < len(X):
                    reshape_size = self.batch_size
                else:
                    reshape_size = len(X) - i
                ybatch = y[idx].reshape((reshape_size, 1))

                ybatch = torch.from_numpy(ybatch).float().cuda()

                output = self.adv_model(Xbatch)
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
        self.adv_model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                Xbatch = torch.from_numpy(Xbatch).float().cuda()
                if len(probas) == 0:
                    probas = self.adv_model(Xbatch).data.cpu().numpy()
                else:
                    probas = np.concatenate((probas, self.adv_model(Xbatch).data.cpu().numpy()), axis=0)
        return probas



    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size
        self.adversarialFunc = params.adversarialFunc
        self.params = params
        sick_advs = {'train': {}, 'dev': {}, 'test': {}}


        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y'])
                                   # key=lambda z: (len(z[0]), len(z[1]), z[2])




            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]


            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)



                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))



        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.sick_data['train']['y'])


        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = self.encode_labels(self.sick_data['test']['y'])

        config = {'seed': self.seed, 'nclasses': 5, 'model_name': params.model_name, 'task_name': self.task_name}
        clf = RelatednessPytorch(train={'X': trainF, 'y': trainY},
                                 valid={'X': devF, 'y': devY},
                                 test={'X': testF, 'y': testY},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config)


        #################################################################################################################


        # devpr, test_yhat = clf.run()
        # print("test yhat shape:")
        # print(test_yhat.shape)
        # pr = pearsonr(test_yhat, self.sick_data['test']['y'])[0]
        # sr = spearmanr(test_yhat, self.sick_data['test']['y'])[0]
        # pr = 0 if pr != pr else pr
        # sr = 0 if sr != sr else sr
        # se = mean_squared_error(test_yhat, self.sick_data['test']['y'])
        # logging.debug('Dev : Pearson {0}'.format(devpr))
        # logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
        #                        for SICK Relatedness\n'.format(pr, sr, se))

        #################################################################################################################


        test_yhat = clf.predict(testF)
        print("test yhat shape:")
        print(test_yhat.shape)
        pr = pearsonr(test_yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(test_yhat, self.sick_data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(test_yhat, self.sick_data['test']['y'])
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                       for SICK Relatedness\n'.format(pr, sr, se))

        train_yhat = clf.predict(trainF)
        print("train yhat shape:")
        print(train_yhat.shape)
        pr = pearsonr(train_yhat, self.sick_data['train']['y'])[0]
        sr = spearmanr(train_yhat, self.sick_data['train']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(train_yhat, self.sick_data['train']['y'])
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                               for SICK Relatedness\n'.format(pr, sr, se))

        dev_yhat = clf.predict(devF)
        print("dev yhat shape:")
        print(dev_yhat.shape)
        pr = pearsonr(dev_yhat, self.sick_data['dev']['y'])[0]
        sr = spearmanr(dev_yhat, self.sick_data['dev']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(dev_yhat, self.sick_data['dev']['y'])
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                                      for SICK Relatedness\n'.format(pr, sr, se))

        y_hat = {'train': {}, 'dev': {}, 'test': {}, 'adv_train': {}, 'adv_dev': {}, 'adv_test': {}}
        y_hat['train'] = train_yhat
        y_hat['test'] = test_yhat
        y_hat['dev'] = dev_yhat


        for key in self.sick_data:
            sick_advs[key]['X_A'] = []
            sick_advs[key]['X_A_orig'] = []
            sick_advs[key]['X_B'] = []
            sick_advs[key]['y'] = []
            sick_advs[key]['sents'] = []
            sick_advs[key]['predicted_y'] = []
            for ii in range(0, len(self.sick_data[key]['X_A']), bsize):

                batch = self.sick_data[key]['X_A'][ii:ii + bsize]
                labels = self.sick_data[key]['y'][ii:ii + bsize]

                embeddings = sick_embed[key]['X_A'][ii:ii + bsize]
                adv_samples, _, new_sentences = self.adversarialFunc(params, batch, labels, embeddings)
                # print(batch[0])
                print("Computing %dth embedding: batch_size %d" % (ii, len(batch)))
                # print(len(adv_samples), bsize)
                if ii + bsize < len(self.sick_data[key]['X_A']):
                    assert len(adv_samples) == bsize

                for sent_adversaries, j in zip(adv_samples, range(len(adv_samples))):
                    b_adversaries = []
                    a_adversaries = []
                    repeated_labels = []
                    predicted_y = []
                    for adv_sample in sent_adversaries:
                        b_adversaries.append(sick_embed[key]['X_B'][ii + j])
                        repeated_labels.append(self.sick_data[key]['y'][ii + j])
                        a_adversaries.append(sick_embed[key]['X_A'][ii + j])
                        predicted_y.append(y_hat[key][ii+j])
                    sick_advs[key]['X_A'].append(sent_adversaries)
                    sick_advs[key]['X_A_orig'].append(a_adversaries)
                    sick_advs[key]['X_B'].append(b_adversaries)
                    sick_advs[key]['y'].append(repeated_labels)
                    sick_advs[key]['sents'].append(new_sentences[j])
                    sick_advs[key]['predicted_y'].append(predicted_y)

            print("no of examples for key:%s:%d,%d" % (key, len(sick_advs[key]['X_A']), len(sick_advs[key]['X_B'])))




        advs_trainA = []
        advs_orig_trainA = []
        advs_trainB = []
        advs_trainY = []
        advs_train_predictedY = []

        for a_advs, b_advs, y_advs, orig_advs, orig_predicted_y in \
                zip(sick_advs['train']['X_A'], sick_advs['train']['X_B'], sick_advs['train']['y'], sick_advs['train']['X_A_orig'], sick_advs['train']['predicted_y']) :
            advs_trainA.extend(a_advs)
            advs_trainB.extend(b_advs)
            advs_trainY.extend(y_advs)
            advs_orig_trainA.extend(orig_advs)
            advs_train_predictedY.extend(orig_predicted_y)
        advs_trainA = np.array(advs_trainA)
        advs_trainB = np.array(advs_trainB)
        advs_trainY = np.array(advs_trainY)
        advs_orig_trainA = np.array(advs_orig_trainA)
        advs_train_predictedY = np.array(advs_train_predictedY)

        print("train adversaries length:%d,%d,%d " % (len(advs_trainA), len(advs_trainB), len(advs_trainY)))
        advs_trainF = np.c_[np.abs(advs_trainA - advs_trainB), advs_trainA * advs_trainB]
        advs_trainY_non_encoded = list(advs_trainY)
        advs_trainY = self.encode_labels(advs_trainY)




        advs_testA = []
        advs_orig_testA = []
        advs_testB = []
        advs_testY = []
        advs_test_predictedY = []
        advs_test_sent_id = []
        sent_id = 0
        for a_advs, b_advs, y_advs, orig_advs, orig_predicted_y in zip(sick_advs['test']['X_A'], sick_advs['test']['X_B'],
                                          sick_advs['test']['y'], sick_advs['test']['X_A_orig'], sick_advs['test']['predicted_y']):
            advs_testA.extend(a_advs)
            advs_testB.extend(b_advs)
            advs_testY.extend(y_advs)
            advs_orig_testA.extend(orig_advs)
            advs_test_predictedY.extend(orig_predicted_y)
            advs_test_sent_id.extend([sent_id]*len(a_advs))
            sent_id+=1

        advs_testA = np.array(advs_testA)
        advs_testB = np.array(advs_testB)
        advs_testY = np.array(advs_testY)
        advs_orig_testA = np.array(advs_orig_testA)
        advs_test_predictedY = np.array(advs_test_predictedY)

        print("test adversaries length:%d,%d,%d " % (len(advs_testA), len(advs_testB), len(advs_testY)))
        advs_testF = np.c_[np.abs(advs_testA - advs_testB), advs_testA * advs_testB]
        advs_testY_non_encoded = list(advs_testY)
        advs_testY = self.encode_labels(advs_testY)



        advs_devA = []
        advs_orig_devA = []
        advs_devB = []
        advs_devY = []
        advs_dev_predictedY = []
        advs_dev_sent_id = []
        sent_id = 0
        for a_advs, b_advs, y_advs, orig_advs, orig_predicted_y in zip(sick_advs['dev']['X_A'],
                                                                       sick_advs['dev']['X_B'],
                                                                       sick_advs['dev']['y'],
                                                                       sick_advs['dev']['X_A_orig'],
                                                                       sick_advs['dev']['predicted_y']):
            advs_devA.extend(a_advs)
            advs_devB.extend(b_advs)
            advs_devY.extend(y_advs)
            advs_orig_devA.extend(orig_advs)
            advs_dev_predictedY.extend(orig_predicted_y)
            advs_dev_sent_id.extend([sent_id]*len(a_advs))
            sent_id += 1

        advs_devA = np.array(advs_devA)
        advs_devB = np.array(advs_devB)
        advs_devY = np.array(advs_devY)
        advs_orig_devA = np.array(advs_orig_devA)
        advs_dev_predictedY = np.array(advs_dev_predictedY)

        print("dev adversaries length:%d,%d,%d " % (len(advs_devA), len(advs_devB), len(advs_devY)))
        advs_devF = np.c_[np.abs(advs_devA - advs_devB), advs_devA * advs_devB]
        advs_devY_non_encoded = list(advs_devY)
        advs_devY = self.encode_labels(advs_devY)







        advs_train_yhat = clf.predict(advs_trainF)

        print("advs train yhat shape:")
        print(advs_train_yhat.shape)
        pr = pearsonr(advs_train_yhat, advs_trainY_non_encoded)[0]
        sr = spearmanr(advs_train_yhat, advs_trainY_non_encoded)[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(advs_train_yhat, advs_trainY_non_encoded)
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                               for SICK Relatedness\n'.format(pr, sr, se))




        advs_test_yhat = clf.predict(advs_testF)

        print("advs test yhat shape:")
        print(advs_test_yhat.shape)
        pr = pearsonr(advs_test_yhat, advs_testY_non_encoded)[0]
        sr = spearmanr(advs_test_yhat, advs_testY_non_encoded)[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(advs_test_yhat, advs_testY_non_encoded)
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                       for SICK Relatedness\n'.format(pr, sr, se))

        advs_dev_yhat = clf.predict(advs_devF)

        print("advs dev yhat shape:")
        print(advs_dev_yhat.shape)
        pr = pearsonr(advs_dev_yhat, advs_devY_non_encoded)[0]
        sr = spearmanr(advs_dev_yhat, advs_devY_non_encoded)[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(advs_dev_yhat, advs_devY_non_encoded)
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                               for SICK Relatedness\n'.format(pr, sr, se))



        train_x = {'advs_x': advs_trainA, 'orig_x': advs_orig_trainA, 'y_hat' : advs_train_yhat, 'predicted_y' : advs_train_predictedY}

        dev_x = {'advs_x': advs_devA, 'orig_x': advs_orig_devA, 'y_hat': advs_dev_yhat,
                   'predicted_y': advs_dev_predictedY}

        test_x = {'advs_x': advs_testA, 'orig_x': advs_orig_testA, 'y_hat': advs_test_yhat,
                 'predicted_y': advs_test_predictedY}

        dev_f = np.c_[
            np.abs(dev_x['advs_x'] - dev_x['orig_x']), dev_x['advs_x'] * dev_x['orig_x'], dev_x['y_hat']]
        test_f = np.c_[
            np.abs(test_x['advs_x'] - test_x['orig_x']), test_x['advs_x'] * test_x['orig_x'], test_x['y_hat']]


        train_y = advs_train_predictedY
        dev_y = advs_dev_predictedY
        test_y = advs_test_predictedY

        self.train_y_pred_model(train_x, dev_x, train_y, dev_y)


        dev_preds = self.predict_proba(dev_f)
        dev_se = mean_squared_error(dev_preds, dev_y)
        print("dev squared error: ", dev_se)

        test_preds = self.predict_proba(test_f)
        print(len(test_f), len(test_preds))
        test_se = mean_squared_error(test_preds, test_y)

        print("test squared error: ", test_se)


        key = 'test'
        test_max_sents = max(advs_test_sent_id)
        new_preds = list(self.sick_data[key]['y'])

        assert len(test_preds) == len(advs_testY_non_encoded)
        print(len(sick_advs[key]['y']), len(advs_testY_non_encoded))
        for i in range(len(test_preds)):
            sent_no = advs_test_sent_id[i]
            #
            if(abs(test_preds[i] - advs_testY_non_encoded[i]) > abs(new_preds[sent_no] - advs_testY_non_encoded[i]) ):
                new_preds[sent_no] = test_preds[i]

        new_preds = np.array(new_preds).reshape((len(new_preds)))

        print("final test yhat shape:")
        print(new_preds.shape)
        pr = pearsonr(new_preds, self.sick_data['test']['y'])[0]
        sr = spearmanr(new_preds, self.sick_data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(new_preds, self.sick_data['test']['y'])
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                                               for SICK Relatedness\n'.format(pr, sr, se))
        test_yhat = test_preds
        devpr = -1

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': test_yhat, 'ndev': len(devA), 'ntest': len(advs_trainA)}

    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j, i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j, i] = np.floor(y) - y + 1
        return Y


class SICKEntailmentEval(SICKRelatednessEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Entailment*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[4])
        sick_data['y'] = [label2id[s] for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.sick_data['test']['y'])

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid}
        clf = SplitClassifier(X={'train': trainF, 'valid': devF, 'test': testF},
                              y={'train': trainY, 'valid': devY, 'test': testY},
                              config=config)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
                       SICK entailment\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(devA), 'ntest': len(testA)}
