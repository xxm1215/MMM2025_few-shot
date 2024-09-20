import liwc
import re
from collections import Counter

# parse 是一个函数，将文本标记（字符串）转换为匹配的 LIWC 类别列表（字符串列表）
# category_names 是词典中的所有 LIWC 类别（字符串列表）

path = 'LIWC2015Dictionary.dic'

def tokenize(text):
    # 简单的分词方法
    return re.findall(r'\w+', text.lower())

# 加载LIWC词典
parse, category_names = liwc.load_token_parser(path)



# 示例文本
text = "Day FourCLEVELAND, OH - JULY 21: Republican presidential candidate Donald Trump gives two thumbs up to the crowd during the evening session on the fourth day of the Republican National Convention on July 21, 2016 at the Quicken Loans Arena in Cleveland, Ohio. Republican presidential candidate Donald Trump received the number of votes needed to secure the party's nomination. An estimated 50,000 people are expected in Cleveland, including hundreds of protesters and members of the media. The four-day Republican National Convention kicked off on July 18. "
# text = "Gop presidential front-runner and billionaire entrepreneur donald trump on thursday unveiled his plan to make the military great again, saying he intends to reinstate the draft as part of a larger effort to bolster americas armed forces. were bringing back the draft, okay? were going to bring it back and were going to make america as strong as we were in the sixties, trump declared while addressing supporters at the pacific amphitheater in costa mesa.  I love the sixties, said trump, continuing, I was a very big supporter of the vietnam war and, of course, the troops… "

# 分词
tokens = tokenize(text)

# 初始化计数器
emotion_count = 0
positive_emotion_count = 0
negative_emotion_count = 0
total_words = len(tokens)
print(total_words)
# 分类和计数
for token in tokens:
    categories = list(parse(token))
    if 'affect' in categories:
        emotion_count += 1
    if 'posemo' in categories:
        positive_emotion_count += 1
    if 'negemo' in categories:
        negative_emotion_count += 1

# 计算整体情感和负面情感的比例
overall_emotion_score = emotion_count / total_words
positive_emotion_score = positive_emotion_count / total_words
negative_emotion_score = negative_emotion_count / total_words

print(f"Overall Emotion Score: {overall_emotion_score}")
print(f"Positive Emotion Score: {positive_emotion_score}")
print(f"Negative Emotion Score: {negative_emotion_score}")

# import liwc
# import pandas as pd

# # 加载LIWC词典
# def load_liwc_dictionary(path):
#     parse, category_names = liwc.load_token_parser(path)
#     return parse, category_names

# # 分析文本中的情感
# def analyze_emotions(text, parse):
#     tokens = text.split()
#     emotion_count = 0
#     positive_emotion_count = 0
#     negative_emotion_count = 0
#     total_tokens = len(tokens)

#     for token in tokens:
#         categories = list(parse(token))
#         if 'affect' in categories:
#             emotion_count += 1
#         if 'posemo' in categories:
#             positive_emotion_count += 1
#         if 'negemo' in categories:
#             negative_emotion_count += 1

#     # 计算情感比例
#     emotion_ratio = emotion_count / total_tokens if total_tokens > 0 else 0
#     positive_ratio = positive_emotion_count / total_tokens if total_tokens > 0 else 0
#     negative_ratio = negative_emotion_count / total_tokens if total_tokens > 0 else 0

#     return emotion_ratio, positive_ratio, negative_ratio

# # 读取CSV文件
# df = pd.read_csv(r"C:\Users\thangWang\Desktop\liwc\gossipcop_multi.csv", header=None, names=['text', 'id', 'label'])

# # 加载LIWC词典
# parse, category_names = load_liwc_dictionary(path)

# # 应用LIWC分析
# emotion_results = []
# for text in df['text']:
#     emotion_results.append(analyze_emotions(text, parse))

# # 将结果转换为DataFrame并合并
# emotions_df = pd.DataFrame(emotion_results, columns=['overall_emotion_score', 'positive_emotion_score', 'negative_emotion_score'])
# df = pd.concat([df, emotions_df], axis=1)

# # 移除原始的text列
# df = df.drop(columns=['text'])

# # 查看结果
# print(df.head())


# # 可选择保存新的DataFrame
# df.to_csv('gossipcop.csv', index=False)
