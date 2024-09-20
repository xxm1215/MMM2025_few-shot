import pandas as pd
import jieba
from collections import defaultdict
import math

# 加载LIWC词典的函数
def load_liwc(filename, encoding='utf-8'):
    import io
    liwc_file = io.open(filename, 'r', encoding=encoding)
    lines = liwc_file.readlines()
    type2name = {}
    word2type = {}
    type2word = defaultdict(list)
    lc = 0
    for i, line in enumerate(lines):
        if '%' in line:
            lc = i
            break
        tmp = line.strip().split()
        type2name[int(tmp[0])] = tmp[1]
    for line in lines[lc + 1:]:
        tmp = line.strip().split()
        word2type[tmp[0]] = list(map(int, tmp[1:]))
        for t in word2type[tmp[0]]:
            type2word[t].append(tmp[0])
    liwc_file.close()
    return type2name, word2type, type2word


type2name, word2type, type2word = load_liwc(r"C:\Users\thangWang\Downloads\sc_liwc.dic")


# import jieba

# def analyze_sentiment(sentence, word2type):
#     # 使用jieba进行分词
#     words = jieba.cut(sentence)
#     # 初始化积极和消极词汇计数
#     positive_count = 0
#     negative_count = 0

#     # 遍历每个词汇，判断其情感类别
#     for word in words:
#         if word in word2type:
#             # 获取当前词汇的所有LIWC类别
#             categories = word2type[word]
#             for category in categories:
#                 if category == 126:  # 积极情感
#                     positive_count += 1
#                 elif category == 127:  # 消极情感及其子类
#                     negative_count += 1

#     # 计算总体情感
#     if positive_count > negative_count:
#         overall_sentiment = "Positive"
#     elif negative_count > positive_count:
#         overall_sentiment = "Negative"
#     else:
#         overall_sentiment = "Neutral"

#     return overall_sentiment, positive_count, negative_count

# # 示例句子
# sentence = "我今天非常高兴，因为我买到了我喜欢的书，但我也有点担心明天的考试。"
# overall_sentiment, positive_count, negative_count = analyze_sentiment(sentence, word2type)

# print("Overall Sentiment:", overall_sentiment)
# print("Positive Words Count:", positive_count)
# print("Negative Words Count:", negative_count)
# import jieba

# # 假设 word2type 已经加载好了
# def analyze_sentiment_score(sentence, word2type):
#     # 使用jieba进行分词
#     words = jieba.cut(sentence)
#     tokens = list(jieba.cut(sentence))
#     total_words = len(tokens)

#     # 初始化计数器
#     emotion_count = 0
#     positive_emotion_count = 0
#     negative_emotion_count = 0

#     # 分类和计数
#     for word in words:
#         if word in word2type:
#             # 获取当前词汇的所有LIWC类别
#             categories = word2type[word]
#             for category in categories:
#                 if category == 126:  # 积极情感
#                     positive_emotion_count += 1
#                 elif category == 127:  # 消极情感及其子类
#                     negative_emotion_count += 1
#     emotion_count = positive_emotion_count + negative_emotion_count

#     # 计算整体情感和积极/消极情感的比例
#     overall_emotion_score = emotion_count / total_words if total_words else 0
#     positive_emotion_score = positive_emotion_count / total_words if total_words else 0
#     negative_emotion_score = negative_emotion_count / total_words if total_words else 0

#     return overall_emotion_score, positive_emotion_score, negative_emotion_score



# data = pd.read_csv(r"G:\WeChat\WeChat Files\wxid_ykajha0raibz22\FileStorage\File\2024-09\label_0_data.csv", header=None)  # 如果没有列名，请使用header=None

# count = 0
# texts = data[0]
# sentence = "我今天非常高兴，因为我买到了我喜欢的书，但我也有点担心明天的考试。"
# total_over = total_pos = total_neg = 0
# squared_over = squared_pos = squared_neg = 0
# for text in data[0]:
#     overall, positive, negative = analyze_sentiment_score(text, word2type)
#     total_over += overall
#     total_pos += positive
#     total_neg += negative
#     squared_over += overall ** 2
#     squared_pos += positive ** 2
#     squared_neg += negative ** 2
#     count += 1

# mean_over = total_over / count
# mean_pos = total_pos / count
# mean_neg = total_neg / count

# std_over = math.sqrt((squared_over / count) - (mean_over ** 2))
# std_pos = math.sqrt((squared_pos / count) - (mean_pos ** 2))
# std_neg = math.sqrt((squared_neg / count) - (mean_neg ** 2))

# print(f"Total Texts Processed: {count}")
# print(f"Overall Emotion Score - Mean: {mean_over}, Std Dev: {std_over}")
# print(f"Positive Emotion Score - Mean: {mean_pos}, Std Dev: {std_pos}")
# print(f"Negative Emotion Score - Mean: {mean_neg}, Std Dev: {std_neg}")

import pandas as pd
import jieba
import numpy as np

def analyze_sentiment_score(sentence, word2type):
    tokens = list(jieba.cut(sentence))
    total_words = len(tokens)

    positive_emotion_count = 0
    negative_emotion_count = 0

    for token in tokens:
        if token in word2type:
            categories = word2type[token]
            for category in categories:
                if category == 126:  # 积极情感
                    positive_emotion_count += 1
                elif category == 127:  # 消极情感及其子类
                    negative_emotion_count += 1

    emotion_count = positive_emotion_count + negative_emotion_count
    overall_emotion_score = emotion_count / total_words if total_words else 0
    positive_emotion_score = positive_emotion_count / total_words if total_words else 0
    negative_emotion_score = negative_emotion_count / total_words if total_words else 0

    return overall_emotion_score, positive_emotion_score, negative_emotion_score

# 加载数据
data = pd.read_csv(r"G:\WeChat\WeChat Files\wxid_ykajha0raibz22\FileStorage\File\2024-09\label_1_data.csv", header=None)  # 修改为实际路径

overall_scores = []
positive_scores = []
negative_scores = []

for text in data[0]:
    scores = analyze_sentiment_score(text, word2type)
    overall_scores.append(scores[0])
    positive_scores.append(scores[1])
    negative_scores.append(scores[2])

# 计算平均值和标准差
overall_mean = np.mean(overall_scores)
positive_mean = np.mean(positive_scores)
negative_mean = np.mean(negative_scores)

overall_std = np.std(overall_scores)
positive_std = np.std(positive_scores)
negative_std = np.std(negative_scores)

print(f"Total Texts Processed: {len(overall_scores)}")
print(f"Overall Emotion Score - Mean: {overall_mean}, Std Dev: {overall_std}")
print(f"Positive Emotion Score - Mean: {positive_mean}, Std Dev: {positive_std}")
print(f"Negative Emotion Score - Mean: {negative_mean}, Std Dev: {negative_std}")
