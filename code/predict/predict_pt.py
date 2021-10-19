#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:01
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *

from model.model_pt import S2S
from data_deal.input_pt import gen_encode_share_position
import torch
import json
import numpy as np

import time


class GenPredict:
    def __init__(self, is_chitchat=True):
        if is_chitchat:
            name = 'chitchat'
        else:
            name = 'kgchat'

        model_cls = S2S(name=name, is_predict=True)
        self.gen_model = model_cls.model
        self.gen_model.to(PT_DEVICE)
        self.gen_model.eval()
        self.gen_tokenizer = model_cls.tokenizer
        self.jieba = None
        self._all_len = []

    @staticmethod
    def check_equal(sample1, sample2):
        if sample1 is None or sample2 is None:
            return False
        if len(sample1[0]) != len(sample2[0]):
            return False
        for s1, s2 in zip(sample1[0], sample2[0]):
            if s1 != s2:
                return False
        return True

    def replace_blank(self, d):
        if d is None:
            return d
        if isinstance(d, list):
            d = [self.replace_blank(_d) for _d in d]
        else:
            assert isinstance(d, str), d
            d = d.replace(' ', '')
        return d

    def get_read_sample(self, file_path):
        with open(file_path, encoding='utf-8') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                line = line.strip()
                sample = json.loads(line)
                history, kg, goal = self.pre_trans(sample)
                yield history, kg, goal
                
    def pre_trans(self, sample:dict):
        kg, goal, history = None, None, sample['history']
        if 'knowledge' in sample.keys() and len(sample['knowledge']) > 0:
            kg = sample['knowledge']
        kg = self.replace_blank(kg)
        if isinstance(kg, list) and len(kg) == 1:
            kg = kg[0]
        if 'goal' in sample.keys() and len(sample['goal']) > 0:
            goal = sample['goal']
        goal = self.replace_blank(goal)
        history = self.replace_blank(history)
        return history, kg, goal

    def encode(self, history, kg, goal):
        token_ids, segs, pos, mask = gen_encode_share_position(
            self.gen_tokenizer, history, kg, goal, need_tip=False, is_predict=True)
        mask = (np.array(mask) > 3) * 1
        return token_ids, segs, pos, mask.tolist()

    def generate(self, history, kg, goal):
        token_ids, segs, pos, mask = self.encode(history, kg, goal)
        with torch.no_grad():
            output = self.gen_model.generate(
                input_ids=torch.Tensor([token_ids]).long().to(PT_DEVICE),
                origin_attention=torch.Tensor([mask]).long().to(PT_DEVICE),
                token_type_ids=torch.Tensor([segs]).long().to(PT_DEVICE),
                position_ids=torch.Tensor([pos]).long().to(PT_DEVICE),
                max_length=config['answer_maxlen'] + len(token_ids),
                eos_token_id=self.gen_tokenizer.token_to_id('[SEP]'),
                num_beams=1,
                # repetition_penalty=1.2,
                # no_repeat_ngram_size=3,
                # do_sample=True,
                # num_return_sequences=3,
            )
        # preds = output.cpu().numpy()[:, len(token_ids):]
        # res = None
        # for pred in preds:
        #     this_res = self.gen_tokenizer.decode(pred).replace(' ', '')
        #     if res is None or len(this_res) < len(res):
        #         res = this_res
        pred = output.cpu().numpy()[0][len(token_ids):]
        res = self.gen_tokenizer.decode(pred)
        return res

    def predict(self, history, kg, goal):
        res = self.generate(history, kg, goal)
        return res

    def predict_file(self, file_path, output_path):
        # 自己确保输出文件夹存在
        print('\n============== {}'.format(file_path))
        with open(output_path, mode='w', encoding='utf-8') as fw:
            i = 0
            last_sample = ''
            res = ''
            for history, kg, goal in self.get_read_sample(file_path):
                i += 1
                # if i > 60:
                #     break
                this_sample = '{}{}{}'.format(history, kg, goal)
                if this_sample == last_sample:
                    fw.write('{}\n'.format(res))
                    continue
                else:
                    last_sample = this_sample
                res = self.predict(history, kg, goal)
                fw.write('{}\n'.format(res))
                if i % 48 == 0:
                    print('\ri: {}'.format(i), end='       ')
                    fw.flush()
            self._all_len.append(i)
            print(f'\nover. num:{i}')


def predict():
    # 验证测试
    predict_cls = GenPredict(is_chitchat=True)
    predict_cls.predict_file(join(DATA_PATH, 'douban', 'test.txt'), join(OUT_PATH, 'douban.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'duconv', 'test.txt'), join(OUT_PATH, 'duconv.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'kdconv', 'test.txt'), join(OUT_PATH, 'kdconv.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'LCCC', 'test.txt'), join(OUT_PATH, 'lccc.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'tencent', 'test.txt'), join(OUT_PATH, 'tencent.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'weibo', 'test.txt'), join(OUT_PATH, 'weibo.txt'))
    print(f'All data {sum(predict_cls._all_len)}')


if __name__ == '__main__':
    predict()