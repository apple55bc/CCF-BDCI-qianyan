#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 20:19
# @Author  : QXTD-LXH
# @Desc    :
from cfg import *
from model.model_consistent import Consistent
from data_deal.input_consistent import ConsInput
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter


tokenizer = Consistent.get_tokenizer()
data_input = ConsInput(tokenizer)
data_input.read_history()
data_loader = torch.utils.data.DataLoader(data_input, num_workers=0, batch_size=36, collate_fn=ConsInput.pad_func)

# 配置一些训练参数
log_steps = 50  # 多少训练步骤验证一次输出
save_steps = 3000  # 多少步骤保存一次
long_save_time = 3600  # 多少秒保存一次额外的模型
train_steps = 332993  # 总共训练多少步骤  这是一个epoch的需要训练的step…………（当batch_size=36）
begin_step = 56400


if __name__ == '__main__':
    model = Consistent()
    model.to(PT_DEVICE)
    writer = SummaryWriter(join(model.save_dir, 'logs'), flush_secs=15)
    if os.path.exists(model.save_path):
        ckpt = torch.load(model.save_path)
        model.load_state_dict(ckpt)
    last_save_time = time.time()

    data_loader = iter(data_loader)
    print(f'Train on: {PT_DEVICE}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    critizen = torch.nn.CrossEntropyLoss(reduction='mean')

    correct = 0
    totle = 0
    losses = 0.0
    history_losses = []
    last_loss = 1000.0

    for step in range(begin_step, train_steps):
        batch_data = next(data_loader)
        for k, v in batch_data.items():
            batch_data[k] = v.to(PT_DEVICE)
        output = model(input_ids=batch_data['input_ids'], token_type_ids=batch_data['token_type_ids'])
        loss = critizen(output, batch_data['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prediction = torch.argmax(output, 1)
        correct += torch.sum(torch.eq(prediction, batch_data['labels']) * 1.0).cpu().item()
        totle += batch_data['labels'].shape[0]
        losses += loss.cpu().item()
        if step % log_steps == 0:
            print(f'Step: {step}, Loss:{losses / totle:.6f}  Acc: {correct / totle * 100:.2f}%  ')
            writer.add_scalar('loss', losses / totle, step)
            writer.add_scalar('accuracy', correct / totle, step)
            history_losses.append(losses / totle)
            totle = 0
            losses = 0
            correct = 0
        if step % save_steps == 0 and step >= 2 * save_steps:
            print(f'Save to ... {model.save_path}')
            torch.save(model.state_dict(), model.save_path)
            data_input.save_history()
            history_losses = history_losses[-30:]
            this_ave_loss = sum(history_losses) / len(history_losses)
            if this_ave_loss < last_loss:
                last_loss = this_ave_loss
                print(f'----> save to best ...  {this_ave_loss}')
                torch.save(model.state_dict(), join(model.save_dir, 'best.pt'))
        if time.time() - last_save_time > long_save_time:
            torch.save(model.state_dict(), join(model.save_dir, 'train-ed-{}.pt'.format(
                time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())))))
            last_save_time = time.time()