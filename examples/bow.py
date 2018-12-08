# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import pandas as pd

sys.path.insert(1,PATH_TO_SENTEVAL)
from AdversarialModels import WordNetSynonym


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def adversarialFunc(params, batch_sentences, batch_labels, embeddings):
    # sentvec = np.multiply(sentvec, params.wvec_dim)
    modified_vecs = []
    repeated_labels = []
    new_sentences = []
    for sent, i in zip(batch_sentences, range(len(batch_sentences))):
        sent_adv_embeddings = []
        sent_adv_labels = []
        sent_adversaries = []
        sentvec = embeddings[i]
        y_label = batch_labels[i]
        if sentvec is None:
            return np.zeros(params.wvec_dim), np.array(y_label)

        new_sent = list(sent)
        sent_adversaries.append(new_sent)
        sent_adv_labels.append(y_label)
        sent_adv_embeddings.append(sentvec)

        new_sent = list(sent)
        sent_adversaries.append(new_sent)
        sent_adv_labels.append(y_label)
        sent_adv_embeddings.append(sentvec)
        # print sent
        # print sentvec ,"\n"
        for word, word_pos in zip(sent, range(len(sent))):
            # print "new word ", word, "-" *80
            new_sentvec = np.array(sentvec)
            if word in params.word_vec:
                # print word, "-" * 30
                # print params.word_vec[word][:20]
                new_sentvec = np.subtract(sentvec, np.true_divide(params.word_vec[word], len(sent)))
                # print "new sent vec ", "-" * 30
                # print new_sentvec[:20]
                word_syns = WordNetSynonym.get_word_synonym(word)
                # print word_syns
                for syn in word_syns:
                    if syn in params.word_vec:

                        if syn == word:
                            continue

                        # print syn, "-"*30
                        # print params.word_vec[syn][:20]
                        sent_adv_embeddings.append(np.add(new_sentvec, np.true_divide(params.word_vec[syn], len(sent))))
                        sent_adv_labels.append(y_label)

                        new_sent = list(sent)
                        new_sent[word_pos] = syn
                        sent_adversaries.append(new_sent)

                        # print "mod sent vec", "-" * 30
                        # print modified_vecs[len(modified_vecs)-1][:20], "\n"
        modified_vecs.append(sent_adv_embeddings)
        repeated_labels.append(np.array(sent_adv_labels))
        new_sentences.append(sent_adversaries)
    # print "modifies vecs size:", len(modified_vecs)
    # repeated_labels = np.array(repeated_labels)
    # print " lable size:", repeated_labels.size
    return modified_vecs, repeated_labels, new_sentences

# Set params for SentEval
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'model_name': 'bow','batch_size': 128}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2, 'cudaEfficient' : True}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare, adversarialFunc=adversarialFunc)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    # transfer_tasks = ['SST2']
    transfer_tasks = ['STSBenchmark']
    # transfer_tasks = ['MRPC']
    results = se.eval(transfer_tasks)
    # adv_results = results['SST2']['adv_results']
    # uneq_df = pd.DataFrame(adv_results['uneq_adversaries'])
    # total_df = pd.DataFrame(adv_results['total_adversaries'])
    # wrong_df = pd.DataFrame(adv_results['wrong_adversaries'])
    # uneq_df_filename = 'uneq_file.csv'
    # total_df_filename = 'total_file.csv'
    # wrong_df_filename = 'wrong_file.csv'
    # uneq_df.to_csv(uneq_df_filename)
    # total_df.to_csv(total_df_filename)
    # wrong_df.to_csv(wrong_df_filename)
    # print(results['task_results'])
