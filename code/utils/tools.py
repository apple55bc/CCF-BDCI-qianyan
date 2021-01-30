#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/6/18 16:32
@File      :tools.py
@Desc      :
"""
import numpy as np


def normalization(ar):
    return ar / np.sqrt(np.sum(np.square(ar), axis=-1, keepdims=True))


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def strip_duplicate(sentence: str, max_len=15, min_len=2):
    while True:
        s = _strip_duplicate(sentence, max_len=max_len, min_len=min_len)
        if s == sentence:
            break
        else:
            sentence = s
    return sentence

def _strip_duplicate(sentence: str, max_len=15, min_len=2):
    if len(sentence) < min_len * 2:
        return sentence.strip()
    result_indexes = np.array([1] * len(sentence))
    for i in range(len(sentence), min_len * 2 - 1, -1):
        # for j in range(min_len, min(int(i / 2), max_len) + 1):
        for j in range(min(int(i / 2), max_len), min_len - 1, -1):
            if result_indexes[i - 2 * j] == 0 and result_indexes[max(i - 2 * j - 1, 0)] == 0:
                break
            if sentence[i - j:i] == sentence[i - 2 * j:i - j]:
                result_indexes[i - 2 * j:i - j] = 0
                break
            if sentence[i - j:i] == sentence[i - 2 * j - 1:i - j - 1]:
                result_indexes[i - 2 * j:i - j + 1] = 0
                break
    res = ''
    for idx, s in zip(result_indexes, sentence):
        if idx > 0:
            res += s
    return res.strip()


def _test():
    print(strip_duplicate('我就是喜欢喜欢你则么办么办啊哈哈哈哈嗯啊嗯嗯啊嗯'))
    print(strip_duplicate(
        '我 也是 ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！ ！'))


if __name__ == '__main__':
    _test()
