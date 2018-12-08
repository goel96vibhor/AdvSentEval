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



            sick_advs[key]['X_A'] = []
            sick_advs[key]['X_B'] = []
            sick_advs[key]['y'] = []
            sick_advs[key]['sents'] = []
            for ii in range(0, len(self.sick_data[key]['X_A']), bsize):

                batch = self.sick_data[key]['X_A'][ii:ii + bsize]
                labels = self.sick_data[key]['y'][ii:ii + bsize]

                embeddings  = sick_embed[key]['X_A'][ii:ii + bsize]
                adv_samples, _, new_sentences = self.adversarialFunc(params, batch, labels, embeddings)
                # print(batch[0])
                print("Computing %dth embedding: batch_size %d" % (ii, len(batch)))
                # print(len(adv_samples), bsize)
                if ii + bsize < len(self.sick_data[key]['X_A']):
                    assert len(adv_samples) == bsize

                for sent_adversaries, j in zip(adv_samples, range(len(adv_samples))):
                    b_adversaries = []
                    repeated_labels = []
                    for adv_sample in sent_adversaries:
                        b_adversaries.append(sick_embed[key]['X_B'][ii + j])
                        repeated_labels.append(self.sick_data[key]['y'][ii + j])

                    sick_advs[key]['X_A'].append(sent_adversaries)
                    sick_advs[key]['X_B'].append(b_adversaries)
                    sick_advs[key]['y'].append(repeated_labels)
                    sick_advs[key]['sents'].append(new_sentences[j])

            print("no of examples for key:%s:%d,%d" % (key, len(sick_advs[key]['X_A']), len(sick_advs[key]['X_B'])))


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

        advs_trainA = []
        advs_trainB = []
        advs_trainY = []

        # advs_trainA = np.stack(sick_advs['train']['X_A'])
        # advs_trainB = np.concatenate(sick_advs['train']['X_B'])
        # advs_trainY = np.concatenate(sick_advs['train']['y'])


        for a_advs, b_advs, y_advs in zip(sick_advs['train']['X_A'], sick_advs['train']['X_B'], sick_advs['train']['y']) :
            advs_trainA.extend(a_advs)
            advs_trainB.extend(b_advs)
            advs_trainY.extend(y_advs)

        advs_trainA = np.array(advs_trainA)
        advs_trainB = np.array(advs_trainB)
        advs_trainY = np.array(advs_trainY)

        print("train adversaries length:%d,%d,%d " % (len(advs_trainA), len(advs_trainB), len(advs_trainY)))
        advs_trainF = np.c_[np.abs(advs_trainA - advs_trainB), advs_trainA * advs_trainB]
        advs_trainY = self.encode_labels(advs_trainY)

        config = {'seed': self.seed, 'nclasses': 5, 'model_name': params.model_name, 'task_name': self.task_name}
        clf = RelatednessPytorch(train={'X': trainF, 'y': trainY},
                                 valid={'X': devF, 'y': devY},
                                 test={'X': testF, 'y': testY},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config)

        devpr, yhat = clf.run()

        print("yhat shape:")
        print(yhat.shape)
        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(yhat, self.sick_data['test']['y'])
        logging.debug('Dev : Pearson {0}'.format(devpr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                       for SICK Relatedness\n'.format(pr, sr, se))

        config = {'seed': self.seed, 'nclasses': 5, 'model_name': params.model_name, 'task_name': self.task_name}
        clf = RelatednessPytorch(train={'X': advs_trainF, 'y': advs_trainY},
                                 valid={'X': devF, 'y': devY},
                                 test={'X': advs_trainF, 'y': advs_trainY},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config)

        devpr, yhat = clf.run()

        print("yhat shape:")
        print(yhat.shape)
        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(yhat, self.sick_data['test']['y'])
        logging.debug('Dev : Pearson {0}'.format(devpr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                               for SICK Relatedness\n'.format(pr, sr, se))

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(devA), 'ntest': len(testA)}

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
