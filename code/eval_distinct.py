#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2020-ECDT) Task2.
This script evaluates the distinct[1] of the submitted model.

reference:

[1] Li, Jiwei, et al. "A diversity-promoting objective function for neural conversation models."
    arXiv preprint arXiv:1510.03055 (2015).

This requires each team to implement the following function:
def gen_response(self, context):
    Return a response given the context.
    Arguments:
    :param context: list, a list of string, dialogue histories.
    :return: string, a response for the context.
"""
import json
import sys
import codecs


def read_dialog(file_path):
    """
    Read dialogs from file
    :param file_path: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with codecs.open(file_path, 'r', 'utf-8') as f:
        content = json.load(f)
    dialogs = []
    for session in content:
        context = [utterance['message'] for utterance in session['messages']]
        for i in range(2, len(context) + 1):
            dialogs.append(context[:i])
    return dialogs

def read_response(file_path):
    """
    Read response from file
    :param file_path: str, file path to the dataset
    :return: dict, a dict of responses of each domain contained in file
    """
    with codecs.open(file_path, 'r', 'utf-8') as f:
        res = json.load(f)
    return res


def eval_distinct(hyps_resp, n=2):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: distinct score for n-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != str:
        print("ERROR, eval_distinct takes in a list of str, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    num = 0
    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(resp[i: i + n])
            num += 1
    return len(ngram) / num

if __name__ == '__main__':
    response = read_response('../output/res.json')
    res = {}
    for domain in ['film', 'music', 'travel']:
        distinct = eval_distinct(response[domain])
        res[domain] = distinct

    print('distinct scores: ', res)
    print('final distinct score: ', sum(res.values()) / 3.0)
