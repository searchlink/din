# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 12:59
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : data_generator.py
# @Software: PyCharm

import random
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class DataInput:
    def __init__(self, file, batch_size):
        self.file = file
        self.batch_size = batch_size
        self.data_set = self.read_file()
        # 打乱顺序
        random.shuffle(self.data_set)

        self.steps = len(self.data_set) // self.batch_size
        if len(self.data_set) % self.batch_size != 0:
            self.steps = self.steps + 1

    def read_file(self):
        '''读取训练集'''
        res = []
        # max_len = 0
        with open(self.file, "r") as f:
            for line in f:
                line = line.strip().split(";")
                hist = line[0].split(" ")   # 商品历史点击序列
                cate = line[1].split(" ")   # 商品历史点击对应的类别序列
                # max_len = max(max_len, len(hist))   # 序列最大长度
                click_next_item = line[2]
                click_next_item_cate = line[3]
                label = line[4]
                res.append([hist, cate, click_next_item, click_next_item_cate, float(label)])
        return res

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data_set)))
            random.shuffle(idxs)

            hist_item, hist_cat, target_item, target_cate, hist_len, b_label = [], [], [], [], [], []
            for i in idxs:
                item = self.data_set[i][0]
                cate = self.data_set[i][1]
                target_i = self.data_set[i][2]
                target_c = self.data_set[i][3]
                len_ = len(self.data_set[i][0])
                label = float(self.data_set[i][4])

                hist_item.append(item)
                hist_cat.append(cate)
                target_item.append(target_i)
                target_cate.append(target_c)
                hist_len.append(len_)
                b_label.append(label)

                if len(hist_item) == self.batch_size:
                    max_len = max(hist_len)
                    hist_item = pad_sequences(hist_item, max_len, padding="post")
                    hist_cat = pad_sequences(hist_cat, max_len, padding="post")

                    yield [np.array(hist_item), np.array(hist_cat), np.array(target_item), np.array(target_cate), np.array(hist_len), np.array(b_label)], None

                    hist_item, hist_cat, target_item, target_cate, hist_len, b_label = [], [], [], [], [], []


class TestData:
    '''单个输入'''
    def __init__(self, file):
        self.file = file
        self.test_set = self.read_file()

    def read_file(self):
        '''读取训练集'''
        res = []
        with open(self.file, "r") as f:
            for line in f:
                line = line.strip().split(";")
                hist = line[0].split(" ")   # 商品历史点击序列
                cate = line[1].split(" ")   # 商品历史点击对应的类别序列
                click_next_item = line[2]
                click_next_item_cate = line[3]
                label = line[4]
                hist_len = len(hist)
                res.append([np.array([hist]), np.array([cate]), np.array([click_next_item]), np.array([click_next_item_cate]), np.array([hist_len]), float(label)])
        return res

