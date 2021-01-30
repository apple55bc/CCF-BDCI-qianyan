#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 23:48
# @Author  : QXTD-LXH
# @Desc    :
from data_deal.input_pt import InputGen
import random


class InputChitchat(InputGen):
    def get_sample(self):
        sample_weights = [0.2, 0.4, 0.4]
        rand_idxes = list(range(len(sample_weights)))
        cusum_weights = []
        w_sum = 0
        for w in sample_weights:
            w_sum += w
            cusum_weights.append(w_sum)
        gen_iter = [
            self._get_tencent(),
            self._get_weibo(),
            self._get_lccc(),
        ]
        while True:
            iter_idx = random.choices(rand_idxes, cum_weights=cusum_weights, k=1)[0]
            # print(iter_idx)
            iter_random = gen_iter[iter_idx]
            sample = next(iter_random)
            if sample['type'] in self._random_answer_data:
                rand_len = random.randint(1, len(sample['history']))
                sample['history'] = sample['history'][:rand_len]
            yield sample