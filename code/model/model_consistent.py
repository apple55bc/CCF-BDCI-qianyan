#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 19:38
# @Author  : QXTD-LXH
# @Desc    :

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


class Consistent(torch.nn.Module):
    def __init__(self, is_predict=False):
        super().__init__()
        config = BertConfig.from_json_file(join(BERT_PATH, 'bert_config.json'))
        self.bert = BertModel(config, add_pooling_layer=True)
        self.tokenizer = self.get_tokenizer()
        if not is_predict:
            load_tf_weights_in_bert(self.bert, tf_checkpoint_path=join(BERT_PATH, 'bert_model.ckpt'), strip_bert=True)
        self.cls = torch.nn.Linear(768, 2)
        self.save_dir = join(MODEL_PATH, 'consistent')
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = join(self.save_dir, 'trained.pt')

    @staticmethod
    def get_tokenizer():
        return BertTokenizer(vocab_file=join(BERT_PATH, 'vocab.txt'), pad_token='[PAD]')

    def forward(self, **inputs):
        bert_output = self.bert(**inputs)[1]  # cls
        bert_output = self.cls(bert_output)
        return bert_output


def test():
    md = Consistent()
    md.to(PT_DEVICE)
    x = md.tokenizer([['啊哈哈你真笨', '我想吃饭饭']], return_tensors='pt', padding=True).to(PT_DEVICE)
    for k, v in x.items():
        print(f'k: {k},  v: {v.shape}  {v}')
    output = md(**x)
    print(f'output: {output.shape}')


if __name__ == '__main__':
    test()