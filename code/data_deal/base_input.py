#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/11/5 22:04
@Author  : Apple QXTD
@File    : base_input.py
@Desc:   :
"""
from cfg import *
import json
import random


class BaseInput:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._random_answer_data = ['douban', 'lccc', 'duconv', 'kdconv']
        self._read_history = [0 for _ in range(6)]
        self._history_save_path = join(MODEL_PATH, 'history.json')

    def read_history(self):
        if os.path.exists(self._history_save_path):
            print('load history: ', self._history_save_path)
            self._read_history = json.load(open(self._history_save_path, encoding='utf-8'))
            print('history: ', self._read_history)
        else:
            'History not exist!'

    def save_history(self):
        if not os.path.isdir(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        json.dump(self._read_history, open(self._history_save_path, mode='w', encoding='utf-8'), ensure_ascii=False)

    def get_sample(self):
        # sample_weights = [0.15, 0.15, 0.1, 0.25, 0.25]
        # sample_weights = [0.03, 0.03, 0.03, 0.45, 0.46]
        sample_weights = [1.0]
        rand_idxes = list(range(len(sample_weights)))
        cusum_weights = []
        w_sum = 0
        for w in sample_weights:
            w_sum += w
            cusum_weights.append(w_sum)
        gen_iter = [
            # self._get_tencent(),
            # self._get_weibo(),
            # self._get_lccc(),
            self._get_duconv(),
            # self._get_kdconv()
        ]
        while True:
            iter_idx = random.choices(rand_idxes, cum_weights=cusum_weights, k=1)[0]
            # print(iter_idx)
            iter_random = gen_iter[iter_idx]
            sample = next(iter_random)
            if sample['type'] in self._random_answer_data:
                if sample['type'] in ['duconv', 'kdconv']:
                    b_i = 0
                else:
                    b_i = 1
                rand_len = random.randint(b_i, len(sample['history']))
                sample['history'] = sample['history'][:rand_len]
            yield sample
            # token_ids, segs, pos, mask = gen_encode_share_position(
            #     self.tokenizer, sample['history'], sample['triples'], sample['goal'], need_tip=False)
            # yield token_ids, segs, pos, mask

    def _get_tencent(self):
        with open(join(DATA_PATH ,'tencent', 'train.txt'), encoding='utf-8') as fr:
            if self._read_history[1] > 0:
                for _ in range(self._read_history[1]):
                    try:
                        fr.readline()
                    except Exception:
                        continue
            while True:
                try:
                    line = fr.readline()
                    self._read_history[1] += 1
                    if not line:
                        fr.seek(0)
                        self._read_history[1] = 0
                        continue
                    line = line.strip()
                    line = json.loads(line)
                except Exception:
                    # 为什么tencent的数据这么多有毛病的
                    continue
                kg = [s.replace(' ', '') for s in line['knowledge']]
                history = [line['history'].replace(' ', ''), line['response'].replace(' ', '')]
                sample = {
                    'history': history.copy(),
                    'name': '',
                    'triples': kg,
                    'goal': None,
                    'type': 'tencent',
                }
                yield sample

    def _get_weibo(self):
        with open(join(DATA_PATH ,'weibo', 'train.txt'), encoding='utf-8') as fr:
            if self._read_history[2] > 0:
                fr.readlines(self._read_history[2])
            while True:
                line = fr.readline()
                self._read_history[2] += 1
                if not line:
                    fr.seek(0)
                    self._read_history[2] = 0
                    continue
                line = line.strip()
                line = json.loads(line)
                history = [line['history'].replace(' ', ''), line['response'].replace(' ', '')]
                sample = {
                    'history': history.copy(),
                    'name': '',
                    'triples': None,
                    'goal': None,
                    'type': 'weibo',
                }
                yield sample

    def _get_lccc(self):
        with open(join(DATA_PATH, 'LCCC', 'LCCD_train.json'), encoding='utf-8') as fr:
            if self._read_history[3] > 0:
                fr.readlines(self._read_history[3])
            while True:
                line = fr.readline()
                self._read_history[3] += 1
                if not line:
                    fr.seek(0)
                    self._read_history[3] = 0
                    continue
                line = line.strip()
                line = json.loads(line)
                history = [s.replace(' ', '') for s in line['conversation']]
                sample = {
                    'history': history.copy(),
                    'name': '',
                    'triples': None,
                    'goal': None,
                    'type': 'lccc',
                }
                yield sample

    def _get_duconv(self):
        with open(join(DATA_PATH, 'duconv', 'train.txt'), encoding='utf-8') as fr:
            if self._read_history[4] > 0:
                fr.readlines(self._read_history[4])
            while True:
                line = fr.readline()
                self._read_history[4] += 1
                if not line:
                    fr.seek(0)
                    self._read_history[4] = 0
                    continue
                line = line.strip()
                line = json.loads(line)

                goal = line['goal']
                goal = [[s.replace(' ', '') for s in _goal] for _goal in goal]
                kg = line['knowledge']
                kg = [[s.replace(' ', '') for s in _kg] for _kg in kg]
                history = [s.replace(' ', '') for s in line['conversation']]
                sample = {
                    'history': history.copy(),
                    'name': '',
                    'triples': kg,
                    'goal': goal,
                    'type': 'duconv',
                }
                yield sample

    def _get_kdconv(self):
        with open(join(DATA_PATH, 'kdconv', 'train.txt'), encoding='utf-8') as fr:
            if self._read_history[5] > 0:
                fr.readlines(self._read_history[5])
            while True:
                line = fr.readline()
                self._read_history[5] += 1
                if not line:
                    fr.seek(0)
                    self._read_history[5] = 0
                    continue
                line = line.strip()
                line = json.loads(line)
                kg = line['knowledge']
                kg = [[s.replace(' ', '') for s in _kg] for _kg in kg]
                history = [s.replace(' ', '') for s in line['conversation']]
                sample = {
                    'history': history.copy(),
                    'name': '',
                    'triples': kg,
                    'goal': None,
                    'type': 'kdconv',
                }
                yield sample


def _test():
    bi = BaseInput(None)
    i = 0
    for v in bi.get_sample():
        print(v)
        print('{} '.format(i), end='')
        i += 1
        if i > 40000:
            break

if __name__ == '__main__':
    _test()