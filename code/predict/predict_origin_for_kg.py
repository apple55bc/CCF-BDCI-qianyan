#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 20:50
# @Author  : QXTD-LXH
# @Desc    :
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfg import *

from model.model_pt import S2S
from data_deal.input_pt import InputGen, gen_encode_share_position
import torch
import json
import random
import numpy as np
import time
jieba = None


def check_equal(sample1, sample2):
    if sample1 is None or sample2 is None:
        return False
    if len(sample1[0]) != len(sample2[0]):
        return False
    for s1, s2 in zip(sample1[0], sample2[0]):
        if s1 != s2:
            return False
    return True


def replace_blank(d):
    global jieba
    if d is None:
        return d
    if isinstance(d, list):
        d = [replace_blank(_d) for _d in d]
    else:
        assert isinstance(d, str), d
        new_words = d.split(' ')
        if len(new_words) > 1:
            for w in new_words:
                try:
                    jieba.add_word(w)
                except OverflowError:
                    continue
        d = d.replace(' ', '')
    return d


def get_read_iter(file_path):
    with open(file_path, encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.strip()
            line = json.loads(line)
            kg, goal, history = None, None, line['history']
            if 'knowledge' in line.keys() and len(line['knowledge']) > 0:
                kg = line['knowledge']
            kg = replace_blank(kg)
            if isinstance(kg, list) and len(kg) == 1:
                kg = kg[0]
            if 'goal' in line.keys() and len(line['goal']) > 0:
                goal = line['goal']
            goal = replace_blank(goal)
            history = replace_blank(history)
            token_ids, segs, pos, mask = gen_encode_share_position(
                tokenizer, history, kg, goal, need_tip=False, is_predict=True)
            mask = (np.array(mask) > 3) * 1
            yield token_ids, segs, pos, mask.tolist()


def predict_file(file_path, output_path):
    global jieba
    # 自己确保输出文件夹存在
    global all_len
    print('\n============== {}'.format(file_path))
    with open(output_path, mode='w', encoding='utf-8') as fw:
        import jieba as jb
        jieba = jb
        i = 0
        last_sample = ''
        res = ''
        for sample in get_read_iter(file_path):
            token_ids, segs, pos, mask = sample
            i += 1
            # if i > 60:
            #     break
            this_sample = '{}'.format(token_ids)
            if this_sample == last_sample:
                fw.write('{}\n'.format(res))
                continue
            else:
                last_sample = this_sample
            output = model.generate(
                input_ids=torch.Tensor([token_ids]).long().to(PT_DEVICE),
                origin_attention=torch.Tensor([mask]).long().to(PT_DEVICE),
                token_type_ids=torch.Tensor([segs]).long().to(PT_DEVICE),
                position_ids=torch.Tensor([pos]).long().to(PT_DEVICE),
                max_length=config['answer_maxlen'] + len(token_ids),
                eos_token_id=model_cls.tokenizer.token_to_id('[SEP]'),
                num_beams=1,
            )
            pred = output.cpu().numpy()[0][len(token_ids):]
            res = tokenizer.decode(pred)
            res = ' '.join(jieba.lcut(res))
            fw.write('{}\n'.format(res))
            if i % 48 == 0:
                print('\ri: {}'.format(i), end='       ')
                fw.flush()
        all_len.append(i)
        print(f'\nover. num:{i}')


if not os.path.isdir(OUT_PATH):
    os.makedirs(OUT_PATH)
is_chitchat = False
if is_chitchat:
    name = 'chitchat'
else:
    name = 'kgchat'

model_cls = S2S(name=name)
model = model_cls.model
model.to(PT_DEVICE)
model.eval()
tokenizer = model_cls.tokenizer
all_len = []
# 验证测试
# predict_file(join(DATA_PATH, 'douban', 'test.txt'), join(OUT_PATH, 'douban.txt'))
# predict_file(join(DATA_PATH, 'duconv', 'test.txt'), join(OUT_PATH, 'duconv.txt'))
# predict_file(join(DATA_PATH, 'kdconv', 'test.txt'), join(OUT_PATH, 'kdconv.txt'))
# predict_file(join(DATA_PATH, 'LCCC', 'test.txt'), join(OUT_PATH, 'lccc.txt'))
# predict_file(join(DATA_PATH, 'tencent', 'test.txt'), join(OUT_PATH, 'tencent.txt'))
# predict_file(join(DATA_PATH, 'weibo', 'test.txt'), join(OUT_PATH, 'weibo.txt'))
predict_file(join(DATA_PATH, 'DuConv', 'test_2.txt'), join(OUT_PATH, 'duconv.txt'))
# predict_file(join(DATA_PATH, 'LCCC', 'test_2.txt'), join(OUT_PATH, 'lccc.txt'))
print(f'All data {sum(all_len)}')