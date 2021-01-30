#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:43
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
from utils.tools import strip_duplicate

out_path = join(OUT_PATH, 'ccf_baidu_dialog_result.txt')
# 43582
name_list = [
    # 'weibo.txt',
    'lccc.txt',
    'duconv.txt',
    # 'kdconv.txt',
    # 'tencent.txt',
    'out_bd.txt',
]
all_len = 0
diff = 0
with open(out_path, mode='w', encoding='utf-8') as fw:
    for name in name_list:
        with open(join(OUT_PATH, name), encoding='utf-8') as fr:
            print(f'name: {name}')
            while True:
                line = fr.readline()
                if not line:
                    break

                if name == 'lccc.txt':
                    line = strip_duplicate(line, max_len=28, min_len=4)
                    if len(line) == 0:
                        diff += 1
                        line = '我 也 是'
                    line += '\n'

                fw.writelines(line)
                all_len += 1
            print(f'len: {all_len}')
    print(f'All len. {all_len}  Should len: 39090 diff: {diff}')