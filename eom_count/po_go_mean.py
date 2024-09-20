# import pandas as pd

# # 假设数据已经保存在CSV文件中
# df = pd.read_csv('politifact.csv')

# # 按label分组
# grouped = df.groupby('label')

# # 计算每个组的均值
# mean_values = grouped[['overall_emotion_score', 'positive_emotion_score', 'negative_emotion_score']].mean()

# # 计算每个组的标准差
# std_devs = grouped[['overall_emotion_score', 'positive_emotion_score', 'negative_emotion_score']].std()

# print("按标签分组的均值：")
# print(mean_values)
# print("\n按标签分组的标准差：")
# print(std_devs)


import liwc
import re
from collections import defaultdict, Counter

# 加载LIWC词典
def load_liwc_dictionary(path):
    parse, category_names = liwc.load_token_parser(path)
    return parse, category_names

# 文本分析函数，同时返回单词分类
def analyze_text(text, parse):
    tokenized = re.findall(r'\w+', text.lower())
    category_words = defaultdict(list)
    category_counts = Counter()
    
    for token in tokenized:
        categories = list(parse(token))
        for category in categories:
            category_words[category].append(token)
        category_counts.update(categories)
    
    return category_counts, category_words

# 主程序
if __name__ == "__main__":
    liwc_path = 'LIWC2015Dictionary.dic'  # 更新为你的词典路径
    parse, category_names = load_liwc_dictionary(liwc_path)
    
    # 示例文本
    # sample_text = "Day FourCLEVELAND, OH - JULY 21: Republican presidential candidate Donald Trump gives two thumbs up to the crowd during the evening session on the fourth day of the Republican National Convention on July 21, 2016 at the Quicken Loans Arena in Cleveland, Ohio. Republican presidential candidate Donald Trump received the number of votes needed to secure the party's nomination. An estimated 50,000 people are expected in Cleveland, including hundreds of protesters and members of the media. The four-day Republican National Convention kicked off on July 18. "
    sample_text = "Gop presidential front-runner and billionaire entrepreneur donald trump on thursday unveiled his plan to make the military great again, saying he intends to reinstate the draft as part of a larger effort to bolster americas armed forces. were bringing back the draft, okay? were going to bring it back and were going to make america as strong as we were in the sixties, trump declared while addressing supporters at the pacific amphitheater in costa mesa.  I love the sixties, said trump, continuing, I was a very big supporter of the vietnam war and, of course, the troops… "
    
    # 分析文本
    results, words_by_category = analyze_text(sample_text, parse)
    
    # 输出整体情感分数
    posemo_score = results.get('posemo', 0)
    negemo_score = results.get('negemo', 0)
    
    print("Positive Emotion Score:", posemo_score)
    print("Negative Emotion Score:", negemo_score)
    print("Words classified as Positive Emotion:", words_by_category.get('posemo', []))
    print("Words classified as Negative Emotion:", words_by_category.get('negemo', []))
