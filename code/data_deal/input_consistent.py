#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 19:51
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
import torch.utils.data
from utils.tools import sequence_padding
from transformers.tokenization_bert import BertTokenizer
import random
import numpy as np
import json


MAX_LEN = 186


def encode_context(history: list, answer: str, tokenizer: BertTokenizer):
    token_ids = [tokenizer.cls_token_id]
    for sentence in history:
        token_ids.extend(tokenizer.encode(sentence)[1:])
    seg_ids = [0] * len(token_ids)
    token_ids.extend(tokenizer.encode(answer)[1:])
    seg_ids += [1] * (len(token_ids) - len(seg_ids))
    if len(token_ids) >= MAX_LEN:
        token_ids = [tokenizer.cls_token_id] + token_ids[-MAX_LEN + 1:]
        seg_ids = seg_ids[-MAX_LEN:]
    return token_ids, seg_ids


class ConsInput(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer: BertTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer.cls_token_id is not None
        self._read_history = [0 for _ in range(1)]
        self._history_save_path = join(MODEL_PATH, 'history_consistent.json')

    def __iter__(self):
        with open(join(DATA_PATH, 'LCCC', 'LCCD_train.json'), encoding='utf-8') as fr:
            last_neg = '这个样本呢比较简单啊哈哈'
            if self._read_history[0] > 0:
                fr.readlines(self._read_history[0])
            while True:
                line = fr.readline()
                self._read_history[0] += 1
                if not line:
                    fr.seek(0)
                    self._read_history[0] = 0
                    continue
                line = line.strip()
                line = json.loads(line)
                history = [s.replace(' ', '') for s in line['conversation']]
                if len(history) <= 1:
                    continue
                answer_idx = random.randint(1, len(history) - 1)
                token_ids, seg_ids = encode_context(history[:answer_idx], history[answer_idx], self.tokenizer)
                yield {'input_ids': np.array(token_ids, dtype=np.int),
                       'token_type_ids': np.array(seg_ids, dtype=np.int),
                       'labels': np.array(1, dtype=np.int),
                       }
                token_ids, seg_ids = encode_context(history[:answer_idx], last_neg, self.tokenizer)
                yield {'input_ids': np.array(token_ids, dtype=np.int),
                       'token_type_ids': np.array(seg_ids, dtype=np.int),
                       'labels': np.array(0, dtype=np.int),
                       }
                last_neg = history[answer_idx]

    @staticmethod
    def pad_func(inputs):
        result = {}
        for k in inputs[0].keys():
            result[k] = []
        for sample in inputs:
            for k, v in sample.items():
                result[k].append(v)
        for k in result.keys():
            if k != 'labels':
                result[k] = torch.Tensor(sequence_padding(result[k], length=None, padding=0)).long()
            else:
                result[k] = torch.Tensor(np.stack(result[k])).long()
        return result

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


"""
data_loader = torch.utils.data.DataLoader(data_input, num_workers=2, batch_size=2, collate_fn=BaseInput.pad_func)
"""
