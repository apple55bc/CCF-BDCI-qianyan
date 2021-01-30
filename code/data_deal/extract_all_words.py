#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 19:02
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
import json
import tqdm


def get_read_sample(file_path):
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
            if isinstance(kg, list) and len(kg) == 1:
                kg = kg[0]
            if 'goal' in line.keys() and len(line['goal']) > 0:
                goal = line['goal']
            yield history, kg, goal


def extract_words():
    words = {}

    def extract(data):
        if isinstance(data, (list, tuple)):
            for d in data:
                extract(d)
        elif isinstance(data, str):
            data = data.split(' ')
            if len(data) > 1:
                for d in data:
                    words[d] = words.get(d, 0) + 1

    def extract_file(file_path):
        print(file_path)
        all_data = list(get_read_sample(file_path))
        for history, kg, goal in tqdm.tqdm(all_data):
            extract(history)
            extract(kg)
            extract(goal)
            
    extract_file(join(DATA_PATH, 'douban', 'test.txt'))
    extract_file(join(DATA_PATH, 'duconv', 'test.txt'))
    extract_file(join(DATA_PATH, 'kdconv', 'test.txt'))
    extract_file(join(DATA_PATH, 'LCCC', 'test.txt'))
    extract_file(join(DATA_PATH, 'tencent', 'test.txt'))
    extract_file(join(DATA_PATH, 'weibo', 'test.txt'))
    all_words = [w for w in list(set(words.keys())) if 1 < len(w) < 10]
    with open(join(MODEL_PATH, 'all_words.txt'), mode='w', encoding='utf-8') as fw:
        fw.writelines('\n'.join(all_words) + '\n')


if __name__ == '__main__':
    extract_words()