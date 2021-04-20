# CCF-BDCI-qianyan
千言多技能对话，包含闲聊、知识对话、推荐对话

* code/bd_chat：推荐对话代码，沿用了LIC-2020 推荐对话的方案。为了支持cuda11……代码强行改成了2.4可以预测的版本，但是训练……貌似跑不起来，转个
[bd-chat-2020链接](https://github.com/apple55bc/bd-chat-2020) 这里可以用tf1.4训练。所有推荐的训练、预测代码都在这里面。
* code/data_deal：数据处理
* code/model/model_consistent：打分重排模型
* code/model/model_pt：生成模型。知识对话的生成模型和闲聊的生成模型其实共用了一套。
* code/recall：
    * model_recall.py：提取、构建句向量索引
    * get_embedding.py：预测使用
* code/train：训练重排模型和生成模型的代码。train_pt.py中，is_chitchat用来控制训练的是哪一种。
* predict：预测闲聊、知识对话的代码。应该只有predict_chitchat.py和predict_origin_for_kg.py有用吧？这是自动评估阶段的代码，不包含召回部分。
* chat.py：人工评估阶段的预测main类。这里面有很多格式兼容代码，忽略它吧。
* server代码可能有重复的，多数是格式兼容所用，没啥重要的。
* data：存放各类聊天数据。参照/data.png的文件结构。

注意不要覆盖vocab.txt 有修改


transformers==3.4.0
torch==1.7.0
