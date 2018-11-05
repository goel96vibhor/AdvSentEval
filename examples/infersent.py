# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import numpy as np

# get models.py from InferSent repo
from models import InferSent
from AdversarialModels import WordNetSynonym

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = 'fasttext/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = 'infersent1.pkl'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def adversarialFunc(params, batch_sentences, batch_labels, embeddings = None):
    # sentvec = np.multiply(sentvec, params.wvec_dim)


    adv_batch_sentences, adv_labels = params.infersent.prepare_adversarial_samples(batch_sentences, batch_labels)

    print("adv samples size %d",len(adv_batch_sentences))

    total_count = sum(len(x) for x in adv_batch_sentences)
    print("sum of sentences called %d", total_count)

    adv_embeddings = []
    for sent_adversaries, i in zip(adv_batch_sentences, range(len(adv_batch_sentences))):

        sentences = [' '.join(s) for s in sent_adversaries]
        sent_adv_embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
        adv_embeddings.append(sent_adv_embeddings)

        if i%10 == 0:
            print("%d sentences done"%(i))


    print("Adv embeddings shape: %s, adv_labels shape",dim(adv_embeddings),dim(adv_labels))
    return adv_embeddings, adv_labels


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'model_name': 'infersent', 'train': False}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)

    # params_senteval['infersent'] = model
    params_senteval['infersent'] = model.cuda()
    se = senteval.engine.SE(params_senteval, batcher, prepare, adversarialFunc=adversarialFunc)
    transfer_tasks = ['SST2']
    results = se.eval(transfer_tasks)
    print(results)
