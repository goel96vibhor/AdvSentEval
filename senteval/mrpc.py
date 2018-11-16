# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
MRPC : Microsoft Research Paraphrase (detection) Corpus
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score
import pickle

class MRPCEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'msr_paraphrase_train.txt'))
        test = self.loadFile(os.path.join(task_path,
                             'msr_paraphrase_test.txt'))
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.mrpc_data['train']['X_A'] + \
                  self.mrpc_data['train']['X_B'] + \
                  self.mrpc_data['test']['X_A'] + self.mrpc_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                mrpc_data['X_A'].append(text[3].split())
                mrpc_data['X_B'].append(text[4].split())
                mrpc_data['y'].append(text[0])

        mrpc_data['X_A'] = mrpc_data['X_A'][1:]
        mrpc_data['X_B'] = mrpc_data['X_B'][1:]
        mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]
        return mrpc_data


    def generate_adv_samples(self, sst_embed_x, sst_embed_y, batcher = None):

        adv_embed_x = []
        adv_embed_y = []
        adv_sentences = []
        # adv_batch_size = self.params.batch_size

        total_samples = len(sst_embed_x)
        # total_samples = 100
        adv_batch_size = total_samples
        for stidx in range(0, total_samples, adv_batch_size):

            batch_a = self.mrpc_data['test']['X_A'][stidx:stidx + adv_batch_size]
            batch_b = self.mrpc_data['test']['X_B'][stidx:stidx + adv_batch_size]

            orig_vector_a = batcher(self.params, batch_a)
            orig_vector_b = batcher(self.params, batch_b)

            debug_embeds = np.c_[np.abs( orig_vector_a - orig_vector_b), orig_vector_a * orig_vector_b]

            print("debug embeds:", debug_embeds[0])
            print("test actual embeds:", sst_embed_x[0])

            print("orig embeddings lengths:",  len(orig_vector_a), len(orig_vector_b))

            batch_labels = self.mrpc_data['test']['y'][stidx:stidx +
                                             adv_batch_size]
            batch_embeds = sst_embed_x[stidx:stidx + adv_batch_size]

            print("Computing adversarial samples for batch: %d no of sentences %d" %(stidx/adv_batch_size, len(batch_a) ))

            modified_vecs, repeated_labels, adv_batch_sentences = self.adversarialFunc(self.params, batch_a, batch_labels, orig_vector_a)

            for sentence_vectors_a, sentence_labels, sentence_adversaries, i in zip(modified_vecs, repeated_labels, adv_batch_sentences, range(len(batch_b)) ):
                repeated_embeds_b = np.tile(orig_vector_b[i],  (len(sentence_vectors_a), 1 ) )
                sentence_adversary_embeds = np.c_[np.abs(sentence_vectors_a - repeated_embeds_b), sentence_vectors_a * repeated_embeds_b]
                # sentence_adversary_embeds = [ np.c_[np.abs(sent_vector_a - orig_vector_b[i]), sent_vector_a * orig_vector_b[i]] for sent_vector_a in sentence_vectors_a]
                adv_embed_x.append(sentence_adversary_embeds)

                adv_embed_y.append(sentence_labels)
                repeated_sentences_b = batch_b[i] * len(sentence_vectors_a)

                if i==0 and stidx ==0:
                    print("printing different lengths", sentence_adversary_embeds.shape, len(adv_embed_x[0]), len(sentence_vectors_a[0]), len(modified_vecs), len(sentence_labels), len(sentence_adversaries) )

                new_sentences = [sent_adv_a + batch_b[i] for sent_adv_a in sentence_adversaries]
                # print new_sentences[0]
                adv_sentences.append(new_sentences)

            print("%d sentences done"%(stidx))
        print("adv_embed length:%d %d"%(len(adv_embed_x), len(adv_embed_y)))
        return adv_embed_x, adv_embed_y, adv_sentences

    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        if (params.train is not None and params.train == False):
            mrpc_embed = {'train': {}, 'test': {}}
        else:
            mrpc_embed = {'train': {}, 'dev': {}, 'test': {}}
        test_file_x_a = 'embeddings/testx_a_' + params.model_name + "_mrpc.csv"
        test_file_x_b = 'embeddings/testx_b_' + params.model_name + "_mrpc.csv"
        test_file_y = 'embeddings/testy_' + params.model_name + "_mrpc.csv"

        train_file_x_a = 'embeddings/trainx_a' + params.model_name + "_mrpc.csv"
        train_file_x_b = 'embeddings/trainx_b' + params.model_name + "_mrpc.csv"
        train_file_y = 'embeddings/trainy_' + params.model_name + "_mrpc.csv"
        self.params = params
        self.adversarialFunc = params.adversarialFunc

        # for key in self.mrpc_data:
        #     logging.info('Computing embedding for {0}'.format(key))
        #     # Sort to reduce padding
        #     text_data = {}
        #     sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
        #                                self.mrpc_data[key]['X_B'],
        #                                self.mrpc_data[key]['y']),
        #                            key=lambda z: (len(z[0]), len(z[1]), z[2]))
        #
        #     text_data['A'] = [x for (x, y, z) in sorted_corpus]
        #     text_data['B'] = [y for (x, y, z) in sorted_corpus]
        #     text_data['y'] = [z for (x, y, z) in sorted_corpus]
        #
        #     for txt_type in ['A', 'B']:
        #         mrpc_embed[key][txt_type] = []
        #         for ii in range(0, len(text_data['y']), params.batch_size):
        #             batch = text_data[txt_type][ii:ii + params.batch_size]
        #             embeddings = batcher(params, batch)
        #             mrpc_embed[key][txt_type].append(embeddings)
        #         mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
        #     mrpc_embed[key]['y'] = np.array(text_data['y'])
        #     logging.info('Computed {0} embeddings'.format(key))
        #
        #
        #
        # pickle.dump(mrpc_embed['test']['A'], open(test_file_x_a, 'wb'))
        # pickle.dump(mrpc_embed['test']['B'], open(test_file_x_b, 'wb'))
        # pickle.dump(mrpc_embed['test']['y'], open(test_file_y, 'wb'))
        #
        # pickle.dump(mrpc_embed['train']['A'], open(train_file_x_a, 'wb'))
        # pickle.dump(mrpc_embed['train']['B'], open(train_file_x_b, 'wb'))
        # pickle.dump(mrpc_embed['train']['y'], open(train_file_y, 'wb'))
        #
        # print("dumped embedding files")

        logging.info("reading files")
        mrpc_embed['test']['A'] = pickle.load(open(test_file_x_a, 'rb'))
        mrpc_embed['test']['B'] = pickle.load(open(test_file_x_b, 'rb'))
        mrpc_embed['test']['y'] = pickle.load(open(test_file_y, 'rb'))

        mrpc_embed['train']['A'] = pickle.load(open(train_file_x_a, 'rb'))
        mrpc_embed['train']['B'] = pickle.load(open(train_file_x_b, 'rb'))
        mrpc_embed['train']['y'] = pickle.load(open(train_file_y, 'rb'))

        # Train
        trainA = mrpc_embed['train']['A']
        trainB = mrpc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = mrpc_embed['train']['y']

        # Test
        testA = mrpc_embed['test']['A']
        testB = mrpc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = mrpc_embed['test']['y']



        print("trainf vector shape",  trainF.shape)

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold,
                  'adversarial_sample_generator': self.generate_adv_samples
                  if self.adversarialFunc is not None else None,
                  'batcher': batcher
                  if batcher is not None else None
                  }

        # X = {'train': {}, 'valid': {}, 'test': {}}
        # y = {'train': {}, 'valid': {}, 'test': {}}
        #
        # for key in mrpc_embed.keys():
        #     X[key] = mrpc_embed.get(key)['X']
        #     y[key] = mrpc_embed.get(key)['y']


        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        params.task_name = "mrpc"
        devacc, testacc, yhat = clf.run(params)
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for MRPC.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
