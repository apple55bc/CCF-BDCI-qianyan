#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_client.py
"""

from __future__ import print_function

import sys
import socket
import importlib
import json

importlib.reload(sys)

SERVER_IP = "121.36.198.149"
SERVER_PORT = 80

def conversation_client(text):
    """
    conversation_client
    """
    mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysocket.connect((SERVER_IP, SERVER_PORT))
    mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096 * 5)
    mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096 * 5)

    mysocket.sendall(text.encode())
    result = mysocket.recv(4096 * 5).decode()

    mysocket.close()

    return result


def main():
    """
    main
    """

    sample = """
    {"goal": [["[1] 寒暄 ( Bot 主动 ， 根据 给定 的 『 聊天 主题 』 寒暄 ， 第一句 问候 要 带 User 名字 ， 聊天 内容 不要 与 『 聊天 时间 』 矛盾 ， 聊天 要 自然 ， 不要 太 生硬 )--> ...... --> [3] 新闻 推荐 ( Bot 主动 ， 推荐   『 杨幂 』   的 新闻   『 杨幂 和 刘恺威 结婚 4 年 频传 情变 ， 在 嘉行 上市 尽调 披露 杨幂 婚姻 状态 、 夜光 剧本 事件 等 风波 后 ， 终于 官宣 。 婚姻 与 爱情 ， 本是 一件 极其 私密 的 事 ， 是 每个 人 的 自由 ， 只要 没 损害 到 公众 利益 ， 就 没 必要 披露 。 女星 家事 没 必要 向 公众 交代 ！ 你 赞同 么 ？ 证实 离婚   杨幂 为何 没 义务 向 大众 说明 她 的 婚姻 ？ 』 ,   User 接受 。 需要 聊 2 轮 ) --> [4] 再见"]], "user_profile": {"职业状态": "工作", "喜欢 的 电影": ["北京童话"], "年龄区间": "18-25", "同意 的 poi": " 小红鱼时尚川菜馆（大润发店）", "拒绝": ["音乐"], "同意 的 美食": " 毛血旺", "喜欢 的 明星": ["杨幂"], "接受 的 电影": ["我是证人", "恋爱中的城市", "怦然星动", "新天生一对"], "居住地": "日照", "姓名": "蔡松林", "喜欢 的 新闻": ["杨幂 的新闻"], "性别": "男", "没有接受 的 电影": ["消失的子弹"]}, "situation": "聊天 时间 : 上午 8 : 00 ， 去 上班 路上     聊天 主题 : 工作 忙", "knowledge": [["蔡松林", "喜欢 的 新闻", "杨幂"], ["杨幂", "新闻", "杨幂 和 刘恺威 结婚 4 年 频传 情变 ， 在 嘉行 上市 尽调 披露 杨幂 婚姻 状态 、 夜光 剧本 事件 等 风波 后 ， 终于 官宣 。 婚姻 与 爱情 ， 本是 一件 极其 私密 的 事 ， 是 每个 人 的 自由 ， 只要 没 损害 到 公众 利益 ， 就 没 必要 披露 。 女星 家事 没 必要 向 公众 交代 ！ 你 赞同 么 ？ 证实 离婚   杨幂 为何 没 义务 向 大众 说明 她 的 婚姻 ？"]], "history": ["[1] 蔡 松林 先生 ， 早上好 ， 现在 在 干嘛 啊 ？", "现在 在 去 公司 的 路上 。", "那 今天 的 工作 还好 吗 ？", "今天 工作 会 很 忙 啊 。", "[2] 那 也 要 劳逸结合 哦 ， 我 陪 你 聊聊天 吧 ， 你 平常 喜欢 看 谁 的 新闻 啊 ？", "我 最 喜欢 杨幂 的 新闻 了 。", "[3] 杨幂 呀 ， 她 和 刘恺威 结婚 四年 了 ， 一直 传出 来说 是 感情 变 了 ， 但是 在 夜光 剧本 事件 等 风波 后 ， 终于 官宣 了 。", "我 感觉 是 这个 事情 没 必要 向 公众 交代 的 。", "是 的 ， 不管 什么 事 不要 打扰到 人家 明星 的 正常 生活 。", "对 ， 明星 也 应该 有 隐私权 啊 。", "嗯 呢 ， 这是 每个 人 的 自由 。", "[4] 是 啊 ， 有事 的话 再 找 你 。 先 上班 了 。"]}
    """
    sample = """
    {"goal": [["[1] 寒暄 ( Bot 主动 ， 根据 给定 的 『 聊天 主题 』 寒暄 ， 第一句 问候 要 带 User 名字 ， 聊天 内容 不要 与 『 聊天 时间 』 矛盾 ， 聊天 要 自然 ， 不要 太 生硬 )--> ...... --> [3] 新闻 推荐 ( Bot 主动 ， 推荐   『 谢娜 』   的 新闻   『 2 月 1 日 ， 张杰 微 吧 公布 喜讯 ， 谢娜 顺序 生 下 双胞胎 女儿 ， “ 恭喜 张杰 当 爸爸 ， 恭喜 恭喜 ！ 双胞胎 ！ ” 更 多 → 恭喜 ！ 谢娜 诞下 双胞胎 女儿 』 ,   User 接受 。 需要 聊 2 轮 ) --> [4] 再见"]], "user_profile": {"职业状态": "学生", "喜欢 的 电影": ["嫁个100分男人"], "年龄区间": "18-25", "同意 的 poi": " 铭苑麻辣香锅-重庆烤鱼", "拒绝": ["音乐"], "同意 的 美食": " 烤鱼", "喜欢 的 明星": ["谢娜"], "接受 的 电影": ["大玩家", "娜娜的玫瑰战争", "当初应该爱你", "虹猫蓝兔火凤凰"], "居住地": "长春", "姓名": "杨轩国", "喜欢 的 新闻": ["谢娜 的新闻"], "性别": "男", "没有接受 的 电影": ["一路狂奔"]}, "situation": "聊天 时间 : 上午 7 : 00 ， 去 上学 路上     聊天 主题 : 考试 好", "knowledge": [["杨轩国", "喜欢 的 新闻", "谢娜"], ["谢娜", "新闻", "2 月 1 日 ， 张杰 微 吧 公布 喜讯 ， 谢娜 顺序 生 下 双胞胎 女儿 ， “ 恭喜 张杰 当 爸爸 ， 恭喜 恭喜 ！ 双胞胎 ！ ” 更 多 → 恭喜 ！ 谢娜 诞下 双胞胎 女儿"]], "history": ["[1] 杨轩国 同学 ， 上午 好 ， 现在 在 干嘛 啊 ？", "我 现在 在 去 学校 的 路上 哦 。", "听 起来 感觉 心情 很 不错 的 样子 呢 。", "是 啊 ， 这次 考试 我考 的 很 好 呢 。", "[2] 那 真是 恭喜 你 了 ， 要 再接再厉 呀 ， 对 了 ， 你 平常 最 喜欢 看 谁 的 新闻 呢 ？", "嗯 呢 ， 我会 的 ， 我 最 喜欢 看 谢娜 的 新闻 了 。"]}
    """
    # sample = """
    # {"profile": {}, "goal": [["START", "王海祥", "浪漫 搭档"], ["浪漫 搭档", "主演", "王海祥"]], "situation": "", "knowledge": [["王海祥", "主要 成就", "2009年 获得 《 龙的传人 》 进步奖"], ["王海祥", "出生 日期", "1982 - 8 - 19"], ["王海祥", "民族", "汉族"], ["王海祥", "性别", "男"], ["王海祥", "职业", "演员"], ["王海祥", "领域", "明星"], ["王海祥", "代表作", "浪漫 搭档"], ["浪漫 搭档", "上映 时间", "2017年1月6日"], ["浪漫 搭档", "时光网 评分", "- 1"], ["浪漫 搭档", "口碑", "口碑 很 差"], ["浪漫 搭档", "类型", "喜剧"], ["浪漫 搭档", "领域", "电影"], ["浪漫 搭档", "主演", "王海祥"], ["王海祥", "描述", "“ 快递 小哥 ”"], ["王海祥", "主要成就", "2009年 获得 《 龙的传人 》 进步奖"], ["王海祥", "身高", "178cm"]], "history": []}
    # """
    # sample = """
    # {"profile": {}, "goal": [[[""]]], "situation": "", "knowledge": [["王海祥", "主要 成就", "2009年 获得 《 龙的传人 》 进步奖"], ["王海祥", "出生 日期", "1982 - 8 - 19"], ["王海祥", "民族", "汉族"], ["王海祥", "性别", "男"], ["王海祥", "职业", "演员"], ["王海祥", "领域", "明星"], ["王海祥", "代表作", "浪漫 搭档"], ["浪漫 搭档", "上映 时间", "2017年1月6日"], ["浪漫 搭档", "时光网 评分", "- 1"], ["浪漫 搭档", "口碑", "口碑 很 差"], ["浪漫 搭档", "类型", "喜剧"], ["浪漫 搭档", "领域", "电影"], ["浪漫 搭档", "主演", "王海祥"], ["王海祥", "描述", "“ 快递 小哥 ”"], ["王海祥", "主要成就", "2009年 获得 《 龙的传人 》 进步奖"], ["王海祥", "身高", "178cm"]], "history": []}
    # """
    # sample = """
    # {"history": ["够 人 就 开 台", "最坑 爹 的 三缺一", "立即 找人 … …", "楷楷 呢 ?"], "goal": [], "profile": {}, "situation": "", "knowledge": []}
    # """
    # sample = """{"profile": ["{"姓名": "张琪航", "性别": "男", "居住地": "保定", "年龄区间": "26-35", "职业状态": "工作", "喜欢 的  明星": ["王力宏"], "喜欢 的 音乐": ["春雨里洗过的太阳"], "喜欢 的 新闻": ["王力宏 的新闻"], "同意 的 美食": " 京酱肉丝", "同意 的 poi": " 金权道韩式自助烤肉火锅", "拒绝": ["电影"], "接受 的 音乐": ["一首简单的歌(Live)", "盖世英雄", "你不知道的事", "好心分手", "不可能错过你"], "没有接受 的 音乐": ["改变自己(Live)", "花田错(Live)"]}"], "situation": ["聊天 时间 : 晚上 20 : 00 ， 在 家里"], "history": ["[ 1 ]   嗨   ，   你   知道   『   一首   简单   的   歌   (   live   )   』     的   音乐   主唱   是   是   谁   吗   ？"], "goal": [[["[1] 问答 ( User 主动 问 『 一首 简单 的 歌 ( Live ) 』   音乐 主唱 ? ， Bot 回答   『 王力宏 』 ， User 满足 并 好评 ) --> ...... --> [4] 播放 音乐 ( Bot 主动 询问 是否 播放 ， User 同意 后 ， Bot 播放   『 KISS   GOODBYE ( Live ) 』 ) --> [5] 再见"]]], "knowledge": [["王力宏", "获奖", "华语 电影 传媒 大奖 _ 观众 票选 最受 瞩目 表现"], ["王力宏", "获奖", "台湾 电影 金马奖 _ 金马奖 - 最佳 原创 歌曲"], ["王力宏", "获奖", "华语 电影 传媒 大奖 _  观众 票选 最受 瞩目 男演员"], ["王力宏", "获奖", "香港电影 金像奖 _ 金像奖 - 最佳 新 演员"], ["王力宏", "出生地", "美国   纽约"], ["王力宏", "简介", "男明星"], ["王力宏", "简介", "很 认真 的 艺人"], ["王力宏", "简介", "一向 严谨"], ["王力宏", "简介", "“ 小将 ”"], ["王力宏", "简介", "好 偶像"], ["王力宏", "体重", "67kg"], ["王力宏", "成就", "全球 流行音乐 金榜 年度 最佳 男歌手"], ["王力宏", "成就", "加拿大 全国 推崇 男歌手"], ["王力宏", "成就", "第 15 届华鼎奖 全球 最佳 歌唱演员 奖"], ["王力宏", "成就", "MTV 亚洲 音乐 台湾 最 受欢迎 男歌手"], ["王力宏", "成就", "两届 金曲奖 国语 男 演唱 人奖"], ["王力宏", "评论", "力宏 必然 是 最 棒 的 ～ ～ ！"], ["王力宏", "评论", "永远 的 FOREVER   LOVE ~ ！ ！"], ["王力宏", "评论", "在 银 幕 的 表演 和 做 娱乐节目 一样 无趣 ， 装 逼成 性"], ["王力宏", "评论", "PERFECT   MR . RIGHT ! !"], ["王力宏", "评论", "有些 歌 一直 唱進 心底 。 。 高學歷 又 有 才 華 。 。"], ["王力宏", "生日", "1976 - 5 - 17"], ["王力宏", "身高", "180cm"], ["王力宏", "星座", "金牛座"], ["王力宏", "血型", "O型"], ["王力宏", "演唱", "一首 简单 的 歌 ( Live )"], ["王力宏", "演唱", "KISS   GOODBYE ( Live )"], ["一首简单的歌(Live)", "评论", "一首 简单 的 歌 ， 却是 一首 最 不 简单 的 歌 。"], ["一首简单的歌(Live)", "评论", "你 唱 的 也好 好听 ， 是 宝藏 啊 兔 兔"], ["一首简单的歌(Live)", "评论", "超爱 王力宏 的 歌 ， 但是 ， 唱起来 真难 呀 ， 哈哈哈 哈哈哈 ， 这才 是 大神 级别 的 歌手 ！"], ["一首简单的歌(Live)", "评论", "97 年 的 我 ， 不 知 道 是否 有 同道中人 ， 一直 喜欢 这些 歌"], ["一首简单的歌(Live)", "评论", "07 年 那 年初三 ， 第一次 无意 从 同学 手机 中 听到 ， 深深 被 吸引 ， 一直 如此"], ["KISS", "GOODBYE(Live) 评论", "明明 不爱 我 了   为什么 不放过 我"], ["KISS", "GOODBYE(Live) 评论", "得不到 就是 得不到 不要 说 你 不 想要"], ["KISS", "GOODBYE(Live) 评论", "我 知道 你 无意 想 绿 我 ， 只是 忘 了 说 分手 ， 只能 说 我 还是 太嫩 了 ， 没想到 还是 会 被 影响 到 心 情 ， 我 是 真的 深深 被 你 打败 了"], ["KISS GOODBYE(Live)", "评论", "《 Kiss   Goodbye 》 是 一首 朴实无华 、 自然 悦耳 的 抒情歌 ， 歌曲 充分 展现 了 王力宏 自创 的 Chinked - out 音乐风格 的 独特 魅力   。"], ["KISS GOODBYE(Live)", "评论", "《 Kiss   Goodbye 》 是 王氏 情歌 的 催泪 之作 ；"], ["KISS GOODBYE(Live)", "评论", "这 首歌曲 表达 了 恋人 每 一次 的 分离 都 让 人 难以 释怀 ， 每 一次 “ Kiss   Goodbye ” 都 让 人 更 期待 下 一次 的 相聚 。"], ["KISS GOODBYE(Live)", "评论", "王力宏 在 这 首歌 里 写出 了 恋人们 的 心声 ， 抒发 了 恋人 之间 互相 思念 对方 的 痛苦 。"]]}"""
    sample = json.dumps(
{
   "profile": {},
   "situation": "",
   "history": [
      "我 最近 看 了 一部 电影 ， 是 一部 很好看 的 电影 。",
      "是 吗 ， 我 都 很久 没 看到 好看 的 电影 了",
      "我 也是 ， 这部 电影 的 主演 是 钱德勒 · 坎特布瑞 ， 你 知道 吗",
      "知道 ， 著名 演员 呢",
      "我 还 知道 他 是 美国 休斯顿 人 ， 演技 很好 的 ， 他 的 作品 《 神秘代码 》 也 很好看",
      "嗯嗯 我 之前 也 看 过 这个 ， 很喜欢 呢",
      "我 也是 呢 ， 这部 电影 的 评分 有 7.3分 ， 网上 评论 说 凯奇 的 又 一部 烂片 ， 故事 太 扯淡 了 吧",
      "除了 这个 还有 其他 评价 吗 ？",
      "我 觉得 还好 吧 ， 不过 我 觉得 他 演技 不错 ， 他 演 的 这部 剧 还是 挺 好看 的",
      "那 这部 剧 是 什么 类型 的 呢 ？",
      "我 也 觉得 ， 我 也 觉得 ， 我 也 觉得 。",
      "我 说 这部 剧 是 什么 类型 的 ？",
      "我 也 觉得 ， 我 也 觉得 ， 我 也 觉得 。",
      "那 上映 时间 是 什么 时候 啊 ？"
   ],
   "goal": '["START","神秘 代码","钱德勒 · 坎特布瑞"],["钱德勒 · 坎特布瑞","代表作","神秘 代码"]',
   "knowledge": [
      [
         "神秘 代码",
         "上映 时间",
         "2009年10月30日"
      ],
      [
         "神秘 代码",
         "主演",
         "钱德勒 · 坎特布瑞"
      ],
      [
         "神秘 代码",
         "国家",
         "美国"
      ],
      [
         "神秘 代码",
         "时光网 短评",
         "# 凯奇 经典之作 # 神秘 代码 ！ ！"
      ],
      [
         "神秘 代码",
         "时光网 短评",
         "凯奇 的 又 一部 烂片 ， 故事 太 扯淡 了 吧"
      ],
      [
         "神秘 代码",
         "时光网 评分",
         "7.3"
      ],
      [
         "神秘 代码",
         "是否 上映",
         "已 上映"
      ],
      [
         "神秘 代码",
         "类型",
         "剧情"
      ],
      [
         "神秘 代码",
         "类型",
         "悬疑"
      ],
      [
         "神秘 代码",
         "领域",
         "电影"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "性别",
         "男"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "标签",
         "小鲜肉"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "祖籍",
         "美国 休斯顿"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "职业",
         "演员"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "评论",
         "就是 非常 非常 喜欢 超级 可爱 的"
      ],
      [
         "钱德勒 · 坎特布瑞",
         "领域",
         "明星"
      ]
   ],
   "user_profile": {}
}, ensure_ascii=False)
    response = conversation_client(sample)
    print(response)


def test_2():
    with open('../data/DuRecDial/test_2.txt', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.strip()
            print(conversation_client(line))


if __name__ == '__main__':
    # test_2()
    main()