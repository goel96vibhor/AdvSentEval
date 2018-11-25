# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SST - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier
try:
    import pickle as pickle
except ImportError:
    import cPickle as pickle

class SSTEval(object):


    def __init__(self, task_path, nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_path, 'sentiment-test'))

        # train['X'] = train['X'][:30]
        # train['y'] = train['y'][:30]
        # dev['X'] = dev['X'][:30]
        # dev['y'] = dev['y'][:30]
        # test['X'] = test['X'][:30]
        # test['y'] = test['y'][:30]
        self.sst_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        if(params.train is not None and params.train == False):
            self.sst_data = {'dev': self.sst_data['dev'], 'test': self.sst_data['test']}
            samples = self.sst_data['dev']['X'] + self.sst_data['test']['X']
            logging.info("preparing samples for only test and validation data")
        else:
            samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                  self.sst_data['test']['X']
            logging.info("preparing samples for train, test and validation data")
        return prepare(params, samples)

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

    def generate_adv_samples(self, sst_embed_x, sst_embed_y):

        adv_embed_x = []
        adv_embed_y = []
        adv_sentences = []
        # adv_batch_size = self.params.batch_size

        total_samples = len(sst_embed_x)
        # total_samples = 100
        adv_batch_size = total_samples
        for stidx in range(0, total_samples, adv_batch_size):

            batch = self.sst_data['test']['X'][stidx:stidx + adv_batch_size]
            batch_labels = sst_embed_y[stidx:stidx + adv_batch_size]
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
        if(params.train is not None and params.train == False):
            sst_embed = {'dev': {}, 'test': {}}
        else:
            sst_embed = {'train': {}, 'dev': {}, 'test': {}}
        test_file_x = 'embeddings/testx_' + params.model_name + "_sst.csv"
        test_file_y = 'embeddings/testy_' + params.model_name + "_sst.csv"
        dev_file_x = 'embeddings/devx_' + params.model_name + "_sst.csv"
        dev_file_y = 'embeddings/devy_' + params.model_name + "_sst.csv"
        bsize = params.batch_size
        self.params = params
        self.adversarialFunc = params.adversarialFunc

        for key in self.sst_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.sst_data[key]['X'],
                                     self.sst_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))

            sst_embed[key]['X'] = []
            for ii in range(0, len(self.sst_data[key]['y']), bsize):
                n = len(self.sst_data[key]['y'])/bsize
                # if ((ii/bsize)*100/n) % 10 == 0:
                print("%d percent done out of %d"%( ((ii/bsize)*100/n), len(self.sst_data[key]['y'])))
                batch = self.sst_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                sst_embed[key]['X'].append(embeddings)
                # logging.info('computed batch {0}, out of total {1}'.format(ii,bsize))
            sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))


        pickle.dump(sst_embed['test']['X'], open(test_file_x, 'wb'))
        pickle.dump(sst_embed['test']['y'], open(test_file_y, 'wb'))
        pickle.dump(sst_embed['dev']['X'], open(dev_file_x, 'wb'))
        pickle.dump(sst_embed['dev']['y'], open(dev_file_y, 'wb'))

        logging.info("dumped files")
        # sst_embed['test']['X'] = pickle.load(open(test_file_x, 'rb'))
        # sst_embed['test']['y'] = pickle.load(open(test_file_y, 'rb'))
        # sst_embed['dev']['X'] = pickle.load(open(dev_file_x, 'rb'))
        # sst_embed['dev']['y'] = pickle.load(open(dev_file_y, 'rb'))

        # print "printing to check if wordvecs fored correct\n"
        #
        # for word in self.sst_data['test']['X'][0]:
        #     print word, "-"*30
        #     print params.word_vec[word][:20]
        # print "sent embedding", "-"*30
        # print sst_embed['test']['X'][0][:20]
        # print "\n\n"

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'adversarial_sample_generator': self.generate_adv_samples
                                    if self.adversarialFunc is not None else None
                            }

        X = {'train': {}, 'valid': {}, 'test': {}}
        y = {'train': {}, 'valid': {}, 'test': {}}

        for key in sst_embed.keys():
            X[key] = sst_embed.get(key)['X']
            y[key] = sst_embed.get(key)['y']

        # X = {'train': {},
        #      'valid': sst_embed['dev']['X'],
        #      'test': sst_embed['test']['X']}
        # y = {'train': {},
        #      'valid': sst_embed['dev']['y'],
        #      'test': sst_embed['test']['y']}

        clf = SplitClassifier(X, y, config=config_classifier, test_dataX= self.sst_data['test']['X'], test_dataY= self.sst_data['test']['y'] )
        params.task_name = "sst"
        devacc, testacc, adv_results = clf.run(params)
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            SST {2} classification\n'.format(devacc, testacc, self.task_name))

        results = dict()
        results['task_results'] = {'devacc': devacc, 'acc': testacc,
                'ndev': len(sst_embed['dev']['X']),
                'ntest': len(sst_embed['test']['X'])}

        results['adv_results'] = adv_results
        print("added adv results to pass back")
        return results
