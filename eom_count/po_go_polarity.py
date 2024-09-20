import pandas as pd
import csv
import textblob
#
#
def get_data(path3, output_excel):
    all_data = []
    for line in csv.reader(open(path3, 'r')):
        # response=line[0]
        text = line[0]
        image_id = line[1]
        label = line[2]  # 0-true, 1-fake
        d = {}

        if len(text) > 0:
            d["id"] = image_id
            d["txt"] = text
            d['label'] = int(label)
            # d['response'] =response
            # Calculate emo value
            blob = textblob.TextBlob(text)
            result_sentiment = blob.sentiment
            polarity = result_sentiment.polarity  # 情感极性
            subjectivity = result_sentiment.subjectivity  # 主观
            # emo = polarity + subjectivity
            d['polarity'] = polarity
            d['subjectivity'] = subjectivity
        else:
            continue
        all_data.append(d)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to Excel
    df.to_excel(output_excel, index=False)


# Sample usage
path3 = r"E:\pythonwork\NLP\CCL2024\20240330_ccl\datasets\fakenewsnet\politifact_multi.csv"
output_excel = "politifact_dataset.xlsx"
get_data(path3, output_excel)
# import pandas as pd
#
# #读取CSV文件
# df = pd.read_csv('llm_generatescore_emoscore.csv')
#
# # 提取polarity列
# polarity_column = df['polarity']
# subjectivity_column=df['subjectivity']
# # 归一化
# normalized_polarity = (polarity_column - polarity_column.min()) / (polarity_column.max() - polarity_column.min())
# # 替换原始数据中的polarity列
# df['polarity'] = normalized_polarity
# df['emoscore'] = df['polarity'] + df['subjectivity']
# emoscore_column=df['emoscore']
# normalized_emoscore = (emoscore_column - emoscore_column.min()) / (emoscore_column.max() - emoscore_column.min())
# df['emoscore']=normalized_emoscore
# # 可选择将归一化后的数据写入新的CSV文件
# df.to_csv('llm_poli.csv', index=False)

# 显示结果
# print(df)
# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv(r'C:\Users\22947\Desktop\ccl\normalized_data_emoscore_goss (1).csv')
#
# # 创建两个区间
# interval1 = (0, 0.5)
# interval2 = (0.6, 1)
#
# # 统计在两个区间内的个数
# interval1_counts = df[(df['polarity'] >= 0) & (df['polarity'] <= 0.5) & (df['label'] ==1)].count()
# interval2_counts = df[(df['polarity'] > 0.5) & (df['polarity'] <= 1) & (df['label'] ==1)].count()
#
# # 输出结果
#
#
# print("在区间 {} 内的个数: {}".format(interval1, interval1_counts['polarity']))
# print("在区间 {} 内的个数: {}".format(interval2, interval2_counts['polarity']))

# text="costa mesa, ca.  gop presidential front-runner and billionaire entrepreneur donald trump on thursday unveiled his plan to make the military great again, saying he intends to reinstate the draft as part of a larger effort to bolster americas armed forces. were bringing back the draft, okay? were going to bring it back and were going to make america as strong as we were in the sixties, trump declared while addressing supporters at the pacific amphitheater in costa mesa.  i love the sixties, said trump, continuing, i was a very big supporter of the vietnam war and, of course, the troops. no one supports the troops more than i do. in addition to bringing back military conscription, mr. trump said that if elected, he will enact legislation that will guarantee citizenship to anyone who serves in the armed forces for a minimum of four years. you have people coming into this country, coming over our borders, expecting a free ride. were not going to give it to them, folks. theyre going to have to earn their citizenship.  trump has repeatedly stated that he wont rule out using nuclear weapons against the islamic state. on thursday he expanded on his claims, saying he would instruct the pentagon to begin testing man-portable tactical nuclear weapons to be used by american soldiers on the battlefield. were not going to have unquestionable military dominance if were seen as too weak to use these weapons against our enemies.  rather than use conventional means such as drone strikes or economic sanctions to respond to acts of terrorism or other forms of aggression against the united states, trump vowed to exercise the nuclear option extremely liberall. my feeling is we have these weapons and weve spent a great deal of money developing and improving them over the years; theres absolutely no reason why we shouldnt be using them.  the former reality tv star accused president obama of making the country less safe by employing tact and diplomacy when dealing with foreign leaders. this guys the president of the united states and hes bowing to the chinese. theyre laughing in our faces, okay? theyre laughing in our faces and if we dont do something about it, theyre going to surpass us militarily and economically and then were going to have a huge problem on our hands.  donald trump kicked off the start of his california campaign on thursday where he made a stop in costa mesa, a semi-rural farming community located in orange county. the gop presidential hopeful is scheduled to address californias republican convention on friday."
# blob = textblob.TextBlob(text)
# result_sentiment = blob.sentiment
# polarity = result_sentiment.polarity  # 情感极性
# subjectivity = result_sentiment.subjectivity  # 主观
# print("情感极性：{}".format(polarity))
# print("情感主观性：{}".format(subjectivity))
