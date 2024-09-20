# import pandas as pd
# import jieba
# from cn_sentiment_measures.sentiment_measures import SentimentMeasures

# # 创建情感度量的实例
# sm = SentimentMeasures()

# # 读取CSV文件
# data = pd.read_csv(r"G:\WeChat\WeChat Files\wxid_ykajha0raibz22\FileStorage\File\2024-09\label_1_data.csv", header=None)  # 如果没有列名，请使用header=None

# # 假设情感文本在CSV的第一列
# texts = data[0]

# # 初始化变量以存储情感度量的总和
# total_apd = total_rpd = total_ls = 0
# total_apd_star = total_rpd_star = total_ls_star = 0

# # 计数器，用于计算平均值
# count = 0

# # 循环处理每个句子
# for text in texts:
#     # 使用结巴分词对句子进行分词
#     word_list = list(jieba.cut(text))
    
#     # 计算情感度量
#     apd = sm.APD(word_list)
#     rpd = sm.RPD(word_list)
#     ls = sm.LS(word_list)
#     # apd_star = sm.APD_with_degree(word_list)
#     # rpd_star = sm.RPD_with_degree(word_list)
#     # ls_star = sm.LS_with_degree(word_list)
    
#     # 累加每个度量的值
#     total_apd += apd
#     total_rpd += rpd
#     total_ls += ls
#     # total_apd_star += apd_star
#     # total_rpd_star += rpd_star
#     # total_ls_star += ls_star
    
#     # 增加计数器
#     count += 1

# # 计算平均值
# average_apd = total_apd / count
# average_rpd = total_rpd / count
# average_ls = total_ls / count
# # average_apd_star = total_apd_star / count
# # average_rpd_star = total_rpd_star / count
# # average_ls_star = total_ls_star / count

# # 打印平均情感度量结果
# print("平均情感度量结果：")
# print(count)
# print(f"APD = {average_apd}")
# print(f"RPD = {average_rpd}")
# print(f"LS = {average_ls}")
# # print(f"APD* = {average_apd_star}")
# # print(f"RPD* = {average_rpd_star}")
# # print(f"LS* = {average_ls_star}")


import pandas as pd
import jieba
from cn_sentiment_measures.sentiment_measures import SentimentMeasures

# 创建情感度量的实例
sm = SentimentMeasures()

# 读取CSV文件
data = pd.read_csv(r"G:\WeChat\WeChat Files\wxid_ykajha0raibz22\FileStorage\File\2024-09\label_0_data.csv", header=None)

# 假设情感文本在CSV的第一列
texts = data[0]

# 初始化变量以存储情感度量的总和和平方和
total_apd = total_rpd = total_ls = 0
total_apd_squared = total_rpd_squared = total_ls_squared = 0

# 计数器，用于计算平均值
count = 0

# 循环处理每个句子
for text in texts:
    # 使用结巴分词对句子进行分词
    word_list = list(jieba.cut(text))
    
    # 计算情感度量
    apd = sm.APD(word_list)
    rpd = sm.RPD(word_list)
    ls = sm.LS(word_list)
    
    # 累加每个度量的值
    total_apd += apd
    total_rpd += rpd
    total_ls += ls
    total_apd_squared += apd ** 2
    total_rpd_squared += rpd ** 2
    total_ls_squared += ls ** 2
    
    # 增加计数器
    count += 1

# 计算平均值
average_apd = total_apd / count
average_rpd = total_rpd / count
average_ls = total_ls / count

# 计算标准差
std_apd = (total_apd_squared / count - average_apd ** 2) ** 0.5
std_rpd = (total_rpd_squared / count - average_rpd ** 2) ** 0.5
std_ls = (total_ls_squared / count - average_ls ** 2) ** 0.5

# 打印平均情感度量结果和标准差
print("平均情感度量结果：")
print(f"Total sentences processed: {count}")
print(f"APD - Mean: {average_apd}, Std Dev: {std_apd}")
print(f"RPD - Mean: {average_rpd}, Std Dev: {std_rpd}")
print(f"LS - Mean: {average_ls}, Std Dev: {std_ls}")
