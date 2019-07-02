# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 11:18
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : util.py
# @Software: PyCharm

import random
import numpy as np
import pandas as pd

def convert_to_df(file_path):
    with open(file_path, "r") as f:
        df = {}
        i = 0
        for line in f:
            df[i] = eval(line)
            i = i + 1

        df = pd.DataFrame.from_dict(df, orient='index')
        return df

reviews_df = convert_to_df("./raw_data/reviews_Electronics_5.json")
meta_df = convert_to_df('./raw_data/meta_Electronics.json')

# 选取在review数据中出现在goods_id, 包含的item相同
meta_df = meta_df[meta_df["asin"].isin(reviews_df["asin"].unique())]
reviews_df = reviews_df[reviews_df["asin"].isin(meta_df["asin"].unique())]
meta_df = meta_df.reset_index(drop=True)

# 取部分字段
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
meta_df = meta_df[['asin', 'categories']]

# 按照最后一类来分
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# 构建有序字段和对应索引
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

asin_map, asin_key = build_map(meta_df, "asin")
cate_map, cate_key = build_map(meta_df, "categories")
revi_map, revi_key = build_map(reviews_df, "reviewerID")

# 按照asin进行排序
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

# reviews_df的asin字段进行映射
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])

# reviews_df 按照reviewerID和时间进行排序
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

user_count, item_count, cate_count, example_count = len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# user_count: 192403	 item_count: 63001	    cate_count: 801	     example_count: 1689188

# 商品对应的类别情况
cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

# 构建训练数据，划分测试集和训练集(商品ID 类目ID 用户ID)
train_set, test_set = [], []
# 用户id, 对应的用户的item历史行为记录
for reviewerID, hist in reviews_df.groupby("reviewerID"):
    # 用户购买的product_id
    pos_list = hist["asin"].tolist()  # hist is datframe

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1))
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))

# 打乱数据
random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

