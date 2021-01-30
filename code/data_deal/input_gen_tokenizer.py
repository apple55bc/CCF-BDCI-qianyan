#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2020/7/17 13:28
@File      :input_gen_tokenizer.py
@Desc      :
"""
from cfg import *
import re


replace_dict = {
    "Information": '信息',
    "Budget": '预算',
    "CEO": '执行总裁',
    "Website": '网址',
    "Country": '国籍',
    "Founded": '成立时间',
    "Founder": '创始人',
    "Location": '位置',
    "Album": '音乐专辑',
    "Name": '名称',
    "Next single": '发行',
    "Established": '组建时间',
    "Artist": '艺术家',
    "Length": '长度',
    "Recorded": '发行时间',
    "Type": '发行时间',
    "Next album": '下一张专辑',
    "Years active": '活跃时间',
    "Genre": '流派',
    "Occupation": '职业',
    "Distributor": '经销商',
    "Associated acts": '乐队成员',
    "Producer": '制作人',
    "Birth place": '出生地',
    "Label": '专辑发行公司',
    "Last album": '最后专辑',
    "This album": '所属专辑',
    "Language": '语言',
    "Released": '发行时间',
}

tip = True


def gen_encode_share_position(tokenizer, context, triples=None, goal=None, need_tip=False, is_predict=False):
    # goal: list: [['START', 'A', 'B']...]
    global tip
    # train_model 要多输入一个无用的y_in
    # 默认第一个attr是真实答案
    _max_single_sentence_length = config['max_single_sentence_length']
    _kg_maxlen = config['kg_maxlen']
    _history_maxlen = config['history_maxlen']
    _goal_maxlen = config['goal_maxlen']
    _answer_maxlen = config['answer_maxlen']

    token_ids = []
    segs = []
    mask = []

    if is_predict:
        history = context
    else:
        history = context[:-1]
    for sentence in history:
        t_ids, _ = tokenizer.encode(sentence, max_length=_max_single_sentence_length)
        if len(token_ids) > 0:
            t_ids = t_ids[1:]
        token_ids.extend(t_ids)
        segs.extend(len(t_ids) * [0])
        mask.extend(len(t_ids) * [1])
    token_ids = token_ids[-_history_maxlen:]
    segs = segs[-_history_maxlen:]
    mask = mask[-_history_maxlen:]
    # 添加 kg
    context_len = len(token_ids)
    ATTR = tokenizer.token_to_id('[ATTR]')
    SBJ = tokenizer.token_to_id('[SBJ]')
    KGEND = tokenizer.token_to_id('[KGEND]')
    GOAL = tokenizer.token_to_id('[GOAL]')

    trip_token_ids = []
    trip_segs = []
    trip_mask = []
    if triples is not None:
        triple_len = 0
        for triple in triples:
            if len(triple) == 0:
                continue
            if isinstance(triple, str):
                t_ids = tokenizer.encode(triple[0], max_length=_max_single_sentence_length)[0][1:]
            else:
                t_ids = tokenizer.encode(triple[0], max_length=_max_single_sentence_length)[0][1:-1]
                t_ids += [GOAL]
                attr = replace_dict.get(triple[1], triple[1])  # 尽量替换为中文
                t_ids += tokenizer.encode(attr, max_length=_max_single_sentence_length)[0][1:-1]
                t_ids += [ATTR]
                t_ids += tokenizer.encode(triple[2], max_length=config['gen_max_single_kg_length'])[0][1:-1]
                t_ids += [KGEND]
            if triple_len + len(t_ids) > _kg_maxlen:
                break
            triple_len += len(t_ids)
            trip_token_ids.extend(t_ids)
            trip_segs.extend(len(t_ids) * [1])
            trip_mask.extend(len(t_ids) * [2])
        if len(trip_token_ids) > _kg_maxlen:
            trip_token_ids = trip_token_ids[-_kg_maxlen:]
            trip_segs = trip_segs[-_kg_maxlen:]
            trip_mask = trip_mask[-_kg_maxlen:]
        triple_len = len(trip_token_ids)
    else:
        triple_len = 0
    token_ids.extend(trip_token_ids)
    segs.extend(trip_segs)
    mask.extend(trip_mask)

    goal_token_ids = []
    goal_segs = []
    goal_mask = []
    if goal is not None:
        goal_len = 0
        for triple in goal:
            if isinstance(triple, str):
                t_ids = tokenizer.encode(triple[0], max_length=_max_single_sentence_length)[0][1:]
            else:
                t_ids = tokenizer.encode(triple[0], max_length=_max_single_sentence_length)[0][1:-1]
                t_ids += [SBJ]
                attr = replace_dict.get(triple[1], triple[1])  # 尽量替换为中文
                t_ids += tokenizer.encode(attr, max_length=_max_single_sentence_length)[0][1:-1]
                t_ids += [ATTR]
                t_ids += tokenizer.encode(triple[2], max_length=config['gen_max_single_kg_length'])[0][1:-1]
                t_ids += [KGEND]
            goal_len += len(t_ids)
            goal_token_ids.extend(t_ids)
            goal_segs.extend(len(t_ids) * [2])
            goal_mask.extend(len(t_ids) * [3])
        if len(trip_token_ids) > _goal_maxlen:
            goal_token_ids = goal_token_ids[-_goal_maxlen:]
            goal_segs = goal_segs[-_goal_maxlen:]
            goal_mask = goal_mask[-_goal_maxlen:]
        goal_len = len(goal_token_ids)
    else:
        goal_len = 0
    token_ids.extend(goal_token_ids)
    segs.extend(goal_segs)
    mask.extend(goal_mask)

    top_len = max(context_len, triple_len, goal_len)
    pos = list(range(top_len - context_len + 1, top_len + 1)) + list(range(top_len - triple_len + 1, top_len + 1))
    if goal_len > 0:
        pos.extend(list(range(top_len - goal_len + 1, top_len + 1)))

    # answer
    if not is_predict:
        answer = context[-1].replace(' ', '')
        ans_rp = re.findall('[嗯哈]{3,}', answer)
        for _rp in ans_rp:
            answer = answer.replace(_rp, _rp[:2])
        t_ids = tokenizer.encode(answer, max_length=_answer_maxlen)[0][1:]
        token_ids.extend(t_ids)
        segs.extend(len(t_ids) * [0])
        pos.extend(range(top_len + 1, top_len + len(t_ids) + 1))
        mask.extend(len(t_ids) * [4])

    if need_tip or tip:
        if tip: tip = False
        print('=' * 20)
        print(' '.join(tokenizer.ids_to_tokens(token_ids)))

    return token_ids, segs, pos, mask