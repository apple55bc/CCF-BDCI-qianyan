#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 20:09
# @Author  : QXTD-LXH
# @Desc    :

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfg import *
import numpy as np
from utils.sif import Sentence2Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import joblib
from utils.tools import normalization
from recall.get_embedding import BertEmbed
from annoy import AnnoyIndex
import random
import json
import re


random.seed(2112)
recall_path = join(MODEL_PATH, 'recall')
if not os.path.isdir(recall_path):
    os.makedirs(recall_path)


class RC_CFG(object):
    def __init__(self):
        self.max_seq_len = 128
        self.emd_dim = 768
        self.pca_dim = 88


recall_config = RC_CFG()


def read_qa():
    # 检索所有的问答对
    with open(join(DATA_PATH, 'weibo', 'train.txt'), encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.strip()
            line = json.loads(line)
            question = line['history'].replace(' ', '')
            answer = line['response'].replace(' ', '')
            if random.random() < 0.7:
                yield question, answer
    with open(join(DATA_PATH ,'tencent', 'train.txt'), encoding='utf-8') as fr:
        while True:
            try:
                line = fr.readline()
                if not line:
                    break
                line = line.strip()
                line = json.loads(line)
            except Exception:
                # 为什么tencent的数据这么多有毛病的
                continue
            question = line['history'].replace(' ', '')
            answer = line['response'].replace(' ', '')
            if random.random() < 0.7:
                yield question, answer
    return


def train_step_1(build_num=4000000, batch_size=512):
    model_emb = BertEmbed()
    az_comp = re.compile('[a-zA-Z0-9]+')
    num_comp = re.compile('[0-9]')

    read_iter = read_qa()

    questions = []
    answers = []

    print('calucate sentences ...')

    num = 0
    for question, answer in read_iter:
        if num >= build_num:
            break
        if len(question) < 4:
            continue
        if len(question) > recall_config.max_seq_len:
            continue
        if len(num_comp.findall(answer)) > 0:  # 包含数字的回复全部丢弃
            continue
        questions.append(re.sub(az_comp, '', question))
        answers.append(answer)
        num += 1

    print(f'questions: {questions[:2]}')
    print(f'answers: {answers[:2]}')
    print(f'len: {len(questions)}')
    print('split sentences ...')
    splited_sentences = []
    for doc in questions[:1000000]:
        splited_sentences.append(list(doc))

    print('train gensim ...')
    word_model = Word2Vec(splited_sentences, min_count=1, size=recall_config.emd_dim, iter=0)
    sif_model = Sentence2Vec(word_model, max_seq_len=recall_config.max_seq_len, components=2)
    print('gensim train done .')
    del splited_sentences, word_model

    print('get vecotrs and train pc...')

    # Memory will explode, rewrite the logic here
    sentence_vectors = []
    vec_batch = batch_size
    pca = PCA(n_components=recall_config.pca_dim, whiten=True, random_state=2112)

    pca_n = min(300000, len(questions))
    has_pca_trained = False

    for b_i, e_i in zip(range(0, len(questions), vec_batch), range(vec_batch, len(questions) + vec_batch, vec_batch)):
        sentences_out = model_emb.get_embedding(questions[b_i:e_i])
        splited_sentences = []
        for doc in questions[b_i:e_i]:
            splited_sentences.append(list(doc))
        sentences_out = sif_model.cal_output(splited_sentences, sentences_out)
        if e_i >= pca_n:
            if has_pca_trained:
                sentence_vectors.extend(normalization(pca.transform(sentences_out)))
            else:
                print('Train PCA ... pca_n num: {}'.format(pca_n))
                sentence_vectors.extend(sentences_out)
                pca.fit(np.stack(sentence_vectors[:pca_n]))
                sentence_vectors = list(normalization(pca.transform(np.stack(sentence_vectors))))
                has_pca_trained = True
        else:
            sentence_vectors.extend(sentences_out)
        del sentences_out, splited_sentences
        print('\r  complete one batch. batch_size: {}  percent {:.2f}%'.format(
            vec_batch, (100 * min(len(questions), e_i) / len(questions))), end='')

    print('\nover !')
    sentence_vectors = np.stack(sentence_vectors)

    sentences_emb = sif_model.train_pc(sentence_vectors)
    print(sentences_emb.shape)
    print('train pc over.')

    print('save model')
    joblib.dump(sif_model, os.path.join(recall_path, 'bert_sif.sif'))
    joblib.dump(pca, os.path.join(recall_path, 'bert_pca.pc'))
    json.dump(answers, open(join(recall_path, 'answers.json'), mode='w', encoding='utf-8'),
              ensure_ascii=False, indent=4, separators=(',', ':'))
    np.save(join(recall_path, 'sentences_emb'), sentences_emb)


def train_step_2():
    print('train_step_2  ...')
    final_q_embs = np.load(join(recall_path, 'sentences_emb.npy'))

    annoy_model = AnnoyIndex(recall_config.pca_dim, metric='angular')
    print('add annoy...')
    for i, emb in enumerate(final_q_embs):
        annoy_model.add_item(i, emb)
    print('build annoy...')
    annoy_model.build(88)
    annoy_model.save(join(recall_path, 'annoy.an'))
    print('build over...')


class SearchEMb:
    def __init__(self, top_n=5):
        self.model_emb = BertEmbed()
        self.sif = joblib.load(join(recall_path, 'bert_sif.sif'))
        self.pca = joblib.load(join(recall_path, 'bert_pca.pc'))
        self.answers = json.load(open(join(recall_path, 'answers.json'), encoding='utf-8'))
        self.annoy = AnnoyIndex(recall_config.pca_dim, metric='angular')
        self.annoy.load(join(recall_path, 'annoy.an'))

        self.az_comp = re.compile('[a-zA-Z0-9]+')
        self.top_n = top_n

    def get_recall(self, sentence, top_n=None):
        if top_n is None:
            top_n = self.top_n
        sentence = re.sub(self.az_comp, '', sentence)
        res_indexs, distances = self.annoy.get_nns_by_vector(self.get_emb(sentence), top_n, include_distances=True)
        results = []
        for idx in res_indexs:
            results.append(self.answers[idx])
        return results, distances

    def get_emb(self, sentence):
        assert isinstance(sentence, str)
        sentence = sentence[-recall_config.max_seq_len + 5:]
        vectors = self.model_emb.get_embedding([sentence])
        mid_vectors = self.sif.cal_output([list(sentence)], vectors)
        mid_vectors = normalization(self.pca.transform(mid_vectors))
        return self.sif.predict_pc(mid_vectors)[0]


def _test():
    search_model = SearchEMb()
    s = '我是最厉害的！'
    while True:
        result = search_model.get_recall(s)
        print(result)
        s = input('In: ')


if __name__ == '__main__':
    train_step_1()
    train_step_2()
    # _test()