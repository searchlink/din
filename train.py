# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 8:45
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : train.py
# @Software: PyCharm

from tqdm import tqdm
import numpy as np
from din.model import din
from din.data_generator import DataInput, TestData
from tensorflow.python.keras.callbacks import Callback
import tensorflow.python.keras as keras

# 定义数据集
batch_size = 64
train_D = DataInput(file="./paddle_train.txt", batch_size=batch_size)
test_D = TestData(file="./paddle_test.txt")

# 构造模型
train_model, label_model = din(item_count=63001, cate_count=801, hidden_units=128)

# 定义sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def calc_auc(raw_arr):
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

# 定义回调函数
class Evaluate(Callback):
    '''回调评估和保存模型'''
    def __init__(self):
        self.acc = []
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.acc.append(acc)
        if acc > self.best_acc:
            self.best_acc = acc
            train_model.save_weights("./best_model.weight")
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best_acc))

    def evaluate(self):
        t_count = 0
        score = []  # 记录实际和预测的结果
        # 取一个batch*20的数据
        np.random.shuffle(test_D.test_set)
        batch_valid_data = test_D.test_set[:(batch_size*20)]
        for row in tqdm(batch_valid_data):
            label = row[-1]
            logits = label_model.predict(row[:-1])  # (batch_size, 1)  array([[0.3211818]], dtype=float32)
            pred = sigmoid(logits)[0][0]

            if label > 0.5:
                score.append([0, 1, pred])
            else:
                score.append([1, 0, pred])

        # 计算AUC
        auc = calc_auc(score)
        print("TEST --> auc: {}".format(auc))
        return auc

# 定义
evaluator = Evaluate()

# 定义ModelCheckpoint、EarlyStopping和TensorBoard
### Using tensorboard callbacks
# tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          callbacks=[evaluator],
                          epochs=2)