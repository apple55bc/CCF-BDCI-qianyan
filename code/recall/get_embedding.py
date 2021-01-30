#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 19:41
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
from transformers.modeling_bert import BertModel
from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from utils.pt_func import load_tf_weights_in_bert


class BertEmbed:
    def __init__(self):
        config = BertConfig.from_json_file(join(BERT_PATH, 'bert_config.json'))
        self.tokenizer = BertTokenizer(vocab_file=join(BERT_PATH, 'vocab.txt'))
        self.model = BertModel(config, add_pooling_layer=False)
        load_tf_weights_in_bert(self.model, tf_checkpoint_path=join(BERT_PATH, 'bert_model.ckpt'), strip_bert=True)
        self.model.to(PT_DEVICE)
        self.model.eval()

    def get_embedding(self, sentences):
        x = self.tokenizer(sentences, return_tensors='pt', padding=True).to(PT_DEVICE)
        with torch.no_grad():
            output = self.model(**x)[0]
        return output.cpu().numpy()


def test():
    md = BertEmbed()
    md.get_embedding(['啊哈哈你真笨', '我想吃饭饭'])


if __name__ == '__main__':
    test()