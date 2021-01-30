#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/6/23 16:26
@File      :cfg.py
@Desc      :
"""
import os

os.environ.setdefault('TF_KERAS', '1')
os.environ.setdefault('USE_TORCH', '1')
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import time
import torch

join = os.path.join


MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = join(MAIN_PATH, 'data')
MID_PATH = join(DATA_PATH, 'mid')
MODEL_PATH = join(MAIN_PATH, 'model')
OUT_PATH = join(MAIN_PATH, 'output')
BERT_PATH = join(DATA_PATH, 'roberta')
TRAIN_PATH = join(DATA_PATH, 'train')
VAL_PATH = join(DATA_PATH, 'valid')

PT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

config = {
    'kg_maxlen': 512 - 56 - 8,
    'history_maxlen': 168,
    'goal_maxlen': 168,
    'answer_maxlen': 56,
    'max_single_sentence_length': 32,
    'gen_max_single_kg_length': 128,
}


def get_logger():
    import logging
    import datetime
    from logging.handlers import RotatingFileHandler

    if not os.path.isdir(os.path.join(MAIN_PATH, 'logs')):
        os.makedirs(os.path.join(MAIN_PATH, 'logs'))

    LOG_PATH = os.path.join(MAIN_PATH, 'logs/log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]')
    # logging.basicConfig(format=formatter, level=logging.INFO,
    #                     filemode='w', datefmt='%Y-%m-%d%I:%M:%S %p')
    _logger = logging.getLogger(__name__)

    #  添加日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    _logger.addHandler(console)

    writer = RotatingFileHandler(filename=LOG_PATH, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5, encoding='utf-8')
    writer.setLevel(logging.INFO)
    writer.setFormatter(formatter)

    _logger.addHandler(writer)
    _logger.setLevel(logging.INFO)

    return _logger


def get_time():
    return '  ---  {}'.format(time.ctime().split(' ')[-2])


def make_dir():
    for d in [
        MID_PATH,
        OUT_PATH,
    ]:
        if not os.path.isdir(d):
            os.makedirs(d)


if __name__ == '__main__':
    make_dir()