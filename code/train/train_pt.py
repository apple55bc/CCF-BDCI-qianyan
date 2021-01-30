#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 22:37
# @Author  : QXTD-LXH
# @Desc    :

from cfg import *

from model.model_pt import S2S
from data_deal.input_pt import InputGen, gen_encode_share_position
from data_deal.input_chitchat import InputChitchat
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import time

is_chitchat = True

epoches = 30000
steps_per_epoch = 300
init_epoch = 0
acc_step = 5

# ==============================================================================
if is_chitchat:
    name = 'chitchat'
    input_cls = InputChitchat
else:
    name = 'kgchat'
    input_cls = InputGen

model_cls = S2S(name=name)
data_input = input_cls(model_cls.tokenizer)
if is_chitchat:
    data_input.batch_size = 28
model = model_cls.model
data_input.read_history()
writer = SummaryWriter(join(MAIN_PATH, 'logs', 'writer'), flush_secs=5)

print('Final batch size: {}, epoch: {}'.format(data_input.batch_size, epoches))
last_save_time = time.time()

if __name__ == '__main__':
    print(model.config)
    print('Device: ', PT_DEVICE)
    eva_iter = data_input.get_sample()
    data_iter = data_input.get_generator()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=2e-5, amsgrad=True)
    for ep in range(270, epoches):
        print('\n' + '=' * 30, ' epoch: {}/{}'.format(ep + 1, epoches))
        model.train()
        for step in range(steps_per_epoch):
            X, S, P, M, L = next(data_iter)
            mask = torch.cumsum(M, dim=-1)
            mask = (mask[:, None, :] <= mask[:, :, None]) * 1
            output = model(
                input_ids=X,
                attention_mask=mask,
                token_type_ids=S,
                position_ids=P,
                return_dict=True
            ).logits

            dim_shape = output.shape[-1]
            M = torch.cat([M[:, 1:], torch.zeros((M.shape[0], 1)).to(PT_DEVICE)], dim=1)
            loss = criterion(output.reshape(-1, dim_shape), L.reshape(-1)) * M.reshape(-1)
            loss = loss.mean() / acc_step
            loss.backward()
            writer.add_scalar('loss', loss.item(), ep * steps_per_epoch + step)
            if ((ep + 1) * steps_per_epoch + step + 1) % acc_step == 0:
                opt.step()
                opt.zero_grad()
            if step % 30 == 0:
                print('loss: ', loss.item())
        
        # 验证测试
        data_input.save_history()
        model.eval()
        sample = next(eva_iter)
        if len(sample['history']) == 0:
            continue
        elif len(sample['history']) == 1:
            pred_idx = 0
        elif len(sample['history']) <= 2:
            pred_idx = 1
        else:
            pred_idx = random.randint(1, len(sample['history']) - 1)
        answer = sample['history'][pred_idx]
        this_history = sample['history'][:pred_idx]
        token_ids, segs, pos, mask = gen_encode_share_position(
            data_input.tokenizer, this_history, sample['triples'], sample['goal'], is_predict=True, need_tip=False)
        if len(token_ids) == 0:
            continue
        mask = (np.array(mask) > 3) * 1
        output = model.generate(
            input_ids=torch.Tensor([token_ids]).long().to(PT_DEVICE),
            origin_attention=torch.Tensor([mask]).long().to(PT_DEVICE),
            token_type_ids=torch.Tensor([segs]).long().to(PT_DEVICE),
            position_ids=torch.Tensor([pos]).long().to(PT_DEVICE),
            max_length=config['answer_maxlen'] + len(token_ids),
            eos_token_id=model_cls.tokenizer.token_to_id('[SEP]'),
        )
        pred = output.cpu().numpy()[0][len(token_ids):]
        eva_info = 'context: {}\n'.format('\t'.join(this_history))
        eva_info += 'gold answer: {}\n'.format(answer)
        eva_info += 'p: {}'.format(data_input.tokenizer.decode(pred))
        print(eva_info)
        writer.add_text('gen', eva_info, ep)
        if (ep + 1) * steps_per_epoch % 1200 == 0:
            model_cls.save()
        if time.time() - last_save_time > 3600:
            model_cls.save(join(MODEL_PATH, name, 'train_{}.pt'.format(
                time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())))))
            last_save_time = time.time()
    writer.close()