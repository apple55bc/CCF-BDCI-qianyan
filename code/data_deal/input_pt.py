#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 22:48
# @Author  : QXTD-LXH
# @Desc    :
from data_deal.base_input import *
from data_deal.input_gen_tokenizer import *
from data_deal.input_gen_tokenizer import gen_encode_share_position
from bert4keras_7_5.snippets import sequence_padding
import numpy as np
import torch


class InputGen(BaseInput):
    def __init__(self, tokenizer):
        super(InputGen, self).__init__(tokenizer)
        self.batch_size = 8
        self.num_comp = re.compile('[0-9]')

    def get_generator(self):
        base_iter = self.get_sample()
        X, S, P, M, L = [], [], [], [], []
        while True:
            sample = next(base_iter)
            if len(sample['history']) < 1:
                continue
            if len(sample['history']) >= 2:
                if sample['history'][-1] in sample['history'][-2]:
                    continue
            token_ids, segs, pos, mask = gen_encode_share_position(
                self.tokenizer, sample['history'], sample['triples'], sample['goal'], need_tip=False)
            mask = (np.array(mask) > 3) * 1
            assert len(token_ids) == len(segs)
            assert len(token_ids) == len(pos)
            assert len(token_ids) == len(mask)
            X.append(token_ids)
            S.append(segs)
            P.append(pos)
            M.append(mask)
            L.append(token_ids[1:] + [0])
            if len(X) >= self.batch_size:
                yield tuple([torch.Tensor(sequence_padding(x)).long().to(PT_DEVICE) for x in [X, S, P, M, L]])
                X, S, P, M, L = [], [], [], [], []


def _test():
    from bert4keras_7_5.tokenizers import Tokenizer
    tk = Tokenizer(join(BERT_PATH, 'vocab.txt'))
    data_deal = InputGen(tk)
    data_deal.batch_size = 3
    gen_it = data_deal.get_generator()
    print('gen it:', gen_it)
    for i in range(20):
        print('\n, ', i)
        X = next(gen_it)
        print(X[0][0].shape)
        print(X[0][0])
        print(X[2][0])
        print(X[3][0])


if __name__ == '__main__':
    _test()
