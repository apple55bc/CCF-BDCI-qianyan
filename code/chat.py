#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 22:40
# @Author  : QXTD-LXH
# @Desc    :
from bd_chat.predict.predict_final import FinalPredict
from predict.predict_pt import GenPredict
from predict.predict_chitchat import Chitchat
from utils.tools import strip_duplicate
import json



class Chat:
    def __init__(self, logger):
        self.logger = logger
        self.model_recommend = FinalPredict()
        self.model_kg = GenPredict(is_chitchat=False)
        self.model_chat = Chitchat()

    def chat(self, sample: dict):
        if 'profile' not in sample.keys():
            assert 'user_profile' in sample.keys()
            sample['profile'] = sample['user_profile']
        if 'knowledge' not in sample.keys():
            assert 'knowlege' in sample.keys()
            sample['knowledge'] = sample['knowlege']
        self.strip_list(sample)
        self.strip_list(sample)
        if 'user_profile' not in sample.keys():
            sample['user_profile'] = sample['profile']
        self.logger.info(json.dumps(sample, ensure_ascii=False, indent=3))
        if len(sample['profile']) > 0 and len(sample['goal']) > 0 and isinstance(sample['profile'], dict):
            self.logger.info('Recommend Chat')
            if isinstance(sample['goal'], list):
                sample['goal'] = ''.join(sample['goal'])
            response = self.recommend(sample)
            if response is None:
                self.logger.info('Rec none to chitchat! ')
                response = self.chitchat(sample)
            elif '再见' in response and len(sample['history']) > 5:
                if '再见' not in sample['history'][-1] or '再见' in sample['history'][-2]:
                    self.logger.info('Add chitchat! ')
                    sample['external_answer'] = response
                    response = self.chitchat(sample)
        elif len(sample['knowledge']) > 0 and len(sample['goal']) > 0:
            self.logger.info('knowledge Chat')
            response = self.kg(sample)
            if '我也觉得' in response.replace(' ', ''):
                self.logger.info('Add kg chitchat! ')
                response = self.chitchat(sample)
            elif len(sample['history']) > 8:
                sample['external_answer'] = response
                sample['neg_score'] = 3
                response = self.chitchat(sample)
        elif isinstance(sample['knowledge'], list) and len(sample['knowledge']) > 3:
            self.logger.info('knowledge Chat')
            response = self.kg(sample)
            if '我也觉得' in response.replace(' ', ''):
                self.logger.info('Add kg chitchat! ')
                response = self.chitchat(sample)
            elif len(sample['history']) > 8:
                sample['external_answer'] = response
                sample['neg_score'] = 3
                response = self.chitchat(sample)
        else:
            self.logger.info('chitchat Chat')
            response = self.chitchat(sample)
        response = strip_duplicate(response, min_len=3, max_len=15)
        return response

    def chat2(self, sample: dict):
        if 'knowledge' not in sample.keys():
            assert 'knowlege' in sample.keys()
            sample['knowledge'] = sample['knowlege']
        if sample['subtrack'] == 'recommend':
            response = self.recommend(sample)
        elif sample['subtrack'] == 'knowledge':
            response = self.kg(sample)
        else:
            assert sample['subtrack'] == 'chitchat'
            response = self.chitchat(sample)
        return response

    def recommend(self, sample: dict):
        response = self.model_recommend.predict(sample)
        return response

    def kg(self, sample: dict):
        history, kg, goal = self.model_kg.pre_trans(sample)
        res = self.model_kg.predict(history=history, kg=kg, goal=goal)
        return res

    def chitchat(self, sample: dict):
        history, kg, goal = self.model_chat.pre_trans(sample)
        if 'external_answer' in sample.keys():
            external_answer = sample['external_answer']
        else:
            external_answer = None
        if 'neg_score' in sample.keys():
            res = self.model_chat.predict(history=history, kg=kg,
                                          goal=goal,
                                          external_answer=external_answer,
                                          neg_score=sample['neg_score'])
        else:
            res = self.model_chat.predict(history=history, kg=kg, goal=goal, external_answer=external_answer)
        return res

    def strip_list(self, sample:dict):
        for k in sample.keys():
            sample[k] = self._strip(sample[k])
        if 'history' in sample.keys() and isinstance(sample['history'], str):
            sample['history'] = [sample['history']]
        if 'conversation' in sample.keys() and isinstance(sample['conversation'], str):
            sample['conversation'] = [sample['conversation']]
        if len(sample['knowledge']) == 3 and isinstance(sample['knowledge'][0], str):
            sample['knowledge'] = [sample['knowledge']]
        if isinstance(sample['profile'], str):
            sample['profile'] = json.loads(sample['profile'])
        if isinstance(sample['goal'], str) and len(sample['goal']) < 5:
            try:
                new_goal = json.loads(sample['goal'])
                sample['goal'] = new_goal
            except json.JSONDecodeError:
                pass

    def _strip(self, d):
        while True:
            if isinstance(d, list) and len(d) == 1:
                d = d[0]
            else:
                break
        return d