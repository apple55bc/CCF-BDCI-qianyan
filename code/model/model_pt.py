#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 22:27
# @Author  : QXTD-LXH
# @Desc    :

from cfg import *
from bert4keras_7_5.tokenizers import Tokenizer, load_vocab
import torch
from transformers.modeling_bert import BertLMHeadModel
from transformers.configuration_bert import BertConfig
from utils.pt_func import load_tf_weights_in_bert
import numpy as np


class GenLM(BertLMHeadModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.

                input_ids=X,
                attention_mask=mask,
                token_type_ids=S,
                position_ids=P,
        """
        origin_attention = kwargs['origin_attention']

        attention_mask = torch.cat(
            [torch.zeros_like(origin_attention), origin_attention.new_ones((origin_attention.shape[0],
                                                          input_ids.shape[1] - origin_attention.shape[1]))], dim=-1
        )
        mask_sum = torch.cumsum(attention_mask, dim=-1)
        mask = (mask_sum[:, None, :] <= mask_sum[:, :, None]) * 1
        token_type_ids = kwargs['token_type_ids']
        token_type_ids = torch.cat(
            [token_type_ids, token_type_ids.new_zeros((token_type_ids.shape[0],
                                                          input_ids.shape[1] - token_type_ids.shape[1]))], dim=-1)
        position_ids = kwargs['position_ids']
        position_ids = torch.cat(
            [position_ids, position_ids[:, -1:] +
             torch.arange(1, input_ids.shape[1] - position_ids.shape[1] + 1)[None, :].long().to(input_ids.device)], dim=-1)
        return {"input_ids": input_ids, 'attention_mask': mask,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                }


class S2S(object):
    def __init__(self, is_predict=False, load_path=None, name='gen'):
        self.save_dir = join(MODEL_PATH, name)
        self.save_path = join(self.save_dir, 'trained.pt')
        self.config_path = join(BERT_PATH, 'bert_config.json')
        self.checkpoint_path = join(BERT_PATH, 'bert_model.ckpt')

        token_dict = load_vocab(
            dict_path=join(BERT_PATH, 'vocab.txt'),
            simplified=False,
        )
        self.tokenizer = Tokenizer(token_dict, do_lower_case=True)

        def get_model():
            bert_config = BertConfig.from_json_file(self.config_path)
            bert_config.type_vocab_size = 3
            bert_config.eos_token_id = self.tokenizer.token_to_id('[SEP]')
            model = GenLM(bert_config)
            if not is_predict:
                load_tf_weights_in_bert(model, self.checkpoint_path)
            # model = keras.models.Model(model.inputs, model.outputs)
            return model

        self.model = get_model()
        self.model.to(PT_DEVICE)

        if load_path:
            self.load(load_path)
        elif os.path.exists(self.save_path):
            self.load(self.save_path)
        elif is_predict:
            raise FileExistsError('No predict file of generate model !')

    def load(self, load_path):
        print('Load from init checkpoint {} .'.format(load_path))
        self.model.load_state_dict(torch.load(load_path))

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        print('Save checkpoint {} .'.format(save_path))
        torch.save(self.model.state_dict(), save_path)
