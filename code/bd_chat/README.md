# 依赖
```text
keras==2.3.1
tensorflow-gpu==1.14.0
easydict
nltk==3.5
```
# 输入格式要求：
 * goal字段，需要是和test集 一模一样的格式，即：‘[1] 新闻 点播 ( User 主动 ， User 问 『 吴亦凡 』   的 新闻 ， ……  --> [3] 再见’ 这种
 * 需要 user_profile 字段， 用户画像信息
 * 需要 situation 字段，环境信息
 
# 文件介绍：
 * conversation_client.py: 类似官方的那种client
 * test_client.py: 交互式的client
 * conversation_server.py: 类似官方的服务。
 