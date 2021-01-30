#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/18 19:58
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
from recall.model_recall import SearchEMb
from model.model_consistent import Consistent
from predict.predict_pt import GenPredict
from data_deal.input_consistent import encode_context
from utils.tools import sequence_padding, strip_duplicate
import numpy as np


class Chitchat(GenPredict):
    def __init__(self):
        super().__init__(is_chitchat=True)
        # self.gen_model 是一个nn.Module模块
        self.recall_model = SearchEMb()  # 这是一个类
        self.consistent_model = Consistent(is_predict=True)  # 这是一个nn.Module模块
        self.consistent_model.to(PT_DEVICE)
        # ckpt = torch.load(join(self.consistent_model.save_dir, 'best.pt'))
        ckpt = torch.load(join(self.consistent_model.save_dir, 'trained.pt'))
        self.consistent_model.load_state_dict(ckpt)
        self.consistent_model.eval()
        self._cls_id = self.consistent_model.tokenizer.cls_token_id

    def get_recall(self, history:list):
        results, distances = self.recall_model.get_recall(history[-1])
        return results, distances

    def predict(self, history, kg, goal, external_answer=None, neg_score=0.3):
        generate_result = self.generate(history, kg, goal)
        generate_result = strip_duplicate(generate_result, min_len=3, max_len=15)
        try:
            recall_result, distances = self.recall_model.get_recall(history[-1], top_n=50)
        except ValueError:
            return generate_result
        # 打分
        all_answers = [generate_result] + recall_result
        distances = [1 + max(1.0 - d, 0.0) for d in distances]
        dis_scores = np.array([1.0] + distances)
        if len(generate_result.strip('，。 ')) > 4 or '也是' not in generate_result:
            dis_scores = np.ones_like(dis_scores)
        if external_answer is not None and isinstance(external_answer, str) and len(external_answer) > 0:
            if '别' in history[-1] or '不' in history[-1]:
                neg_score = 0.15
            all_answers.append(external_answer)
            dis_scores = np.concatenate([dis_scores, np.array([neg_score])], axis=0)
        X, S = [], []
        for answer in all_answers:
            token_ids, seg_ids = encode_context(history, answer, self.consistent_model.tokenizer)
            X.append(token_ids)
            S.append(seg_ids)
        X = sequence_padding(X)
        S = sequence_padding(S)
        X = torch.Tensor(sequence_padding(X, length=None, padding=0)).long().to(PT_DEVICE)
        S = torch.Tensor(sequence_padding(S, length=None, padding=0)).long().to(PT_DEVICE)
        with torch.no_grad():
            output = self.consistent_model(input_ids=X, token_type_ids=S)[:, 1]
            output = torch.softmax(output, dim=-1).cpu().numpy()
        output = dis_scores * output
        idx = np.argmax(output)
        res = all_answers[idx]
        return res


def predict():
    # 验证测试
    predict_cls = Chitchat()
    # predict_cls.predict_file(join(DATA_PATH, 'douban', 'test.txt'), join(OUT_PATH, 'douban.txt'))
    predict_cls.predict_file(join(DATA_PATH, 'LCCC', 'test.txt'), join(OUT_PATH, 'lccc.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'tencent', 'test.txt'), join(OUT_PATH, 'tencent.txt'))
    # predict_cls.predict_file(join(DATA_PATH, 'weibo', 'test.txt'), join(OUT_PATH, 'weibo.txt'))
    print(f'All data {sum(predict_cls._all_len)}')


if __name__ == '__main__':
    predict()