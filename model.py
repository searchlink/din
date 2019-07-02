# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 8:43
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : model.py
# @Software: PyCharm

import random
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

# 设置随机种子，方便复现
seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


class Attention(keras.layers.Layer):
    def __init__(self, attention_hidden_units=(80, 40, 1), attention_activation="sigmoid", supports_masking=True):
        super(Attention, self).__init__()
        self.attention_hidden_units = attention_hidden_units
        self.attention_activation = attention_activation
        self.supports_masking = supports_masking

    def build(self, input_shape):
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        i_emb:     [Batch_size, Hidden_units]
        hist_emb:        [Batch_size, max_len, Hidden_units]
        hist_len: [Batch_size]
        '''
        assert len(x) == 3

        i_emb, hist_emb, hist_len = x[0], x[1], x[2]
        hidden_units = K.int_shape(hist_emb)[-1]
        max_len = tf.shape(hist_emb)[1]

        i_emb = tf.tile(i_emb, [1, max_len])  # (batch_size, max_len * hidden_units)
        i_emb = tf.reshape(i_emb, [-1, max_len, hidden_units])  # (batch_size, max_len, hidden_units)
        concat = K.concatenate([i_emb, hist_emb, i_emb - hist_emb, i_emb * hist_emb],
                               axis=2)  # (batch_size, max_len, hidden_units * 4)

        for i in range(len(self.attention_hidden_units)):
            activation = None if i == 2 else self.attention_activation
            outputs = keras.layers.Dense(self.attention_hidden_units[i], activation=activation)(concat)
            concat = outputs

        outputs = tf.reshape(outputs, [-1, 1, max_len])  # (batch_size, 1, max_len)

        if self.supports_masking:
            mask = tf.sequence_mask(hist_len, max_len)  # (batch_size, 1, max_len)
            padding = tf.ones_like(outputs) * (-1e12)  
            outputs = tf.where(mask, outputs, padding)

        # 对outputs进行scale
        outputs = outputs / (hidden_units ** 0.5)
        outputs = K.softmax(outputs)  


        outputs = tf.matmul(outputs, hist_emb)  # batch_size, 1, hidden_units)

        outputs = tf.squeeze(outputs)  # (batch_size, hidden_units)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def share_weights(hidden_units=63930):
    '''
    reuse a group of keras layers(封装多层，同时可以共享)
    '''
    layers_units = (80, 40, 1)
    share_input = keras.layers.Input(shape=(hidden_units, ))
    share_layer = share_input
    for i in range(len(layers_units)):
        activation = None if i == 2 else "sigmoid"
        share_layer = keras.layers.Dense(layers_units[i], activation=activation)(share_layer)
    out_layer = share_layer
    model = keras.models.Model(share_input, out_layer)
    return model


def din(item_count, cate_count, hidden_units=128):
    '''
    :param item_count: 商品数
    :param cate_count: 类别数
    :param hidden_units: 隐藏单元数
    :return: model
    '''
    target_item = keras.layers.Input(shape=(1,), name='target_item', dtype="int32")  # 点击的item
    target_cate = keras.layers.Input(shape=(1,), name='target_cate', dtype="int32")  # 点击的item对应的所属类别
    label = keras.layers.Input(shape=(1,), name='label', dtype="float32")  # 是否点击

    hist_item_seq = keras.layers.Input(shape=(None,), name="hist_item_seq", dtype="int32")  # 点击序列
    hist_cate_seq = keras.layers.Input(shape=(None,), name="hist_cate_seq", dtype="int32")  # 点击序列对应的类别序列

    hist_len = keras.layers.Input(shape=(1,), name='hist_len', dtype="int32")  # 序列本来的长度

    item_emb = keras.layers.Embedding(input_dim=item_count,
                                      output_dim=hidden_units // 2,
                                      embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4,
                                                                                             seed=seed))
    cate_emb = keras.layers.Embedding(input_dim=cate_count,
                                      output_dim=hidden_units // 2,
                                      embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4,
                                                                                             seed=seed))
    item_b = keras.layers.Embedding(input_dim=item_count, output_dim=1,
                                    embeddings_initializer=keras.initializers.Constant(0.0))

    # get target bias embedding
    target_item_bias_emb = item_b(target_item)  # (batch_size, 1, 1)
    #
    target_item_bias_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(target_item_bias_emb)

    # get target embedding
    target_item_emb = item_emb(target_item)  # (batch_size, 1, hidden_units//2)
    target_cate_emb = cate_emb(target_cate)  # (batch_size, 1, hidden_units//2)
    i_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [target_item_emb, target_cate_emb])  # (batch_size, 1, hidden_units)
    i_emb = keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(i_emb)  # (batch_size, hidden_units)

    # get history item embedding
    hist_item_emb = item_emb(hist_item_seq)  # (batch_size, max_len, hidden_units//2)
    hist_cate_emb = cate_emb(hist_cate_seq)  # (batch_size, max_len, hidden_units//2)
    hist_emb = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))(
        [hist_item_emb, hist_cate_emb])  # (batch_size, max_len, hidden_units)

    # 构建点击序列与候选的attention关系
    din_attention = Attention()([i_emb, hist_emb, hist_len])  # (batch_size, hidden_units)
    din_attention = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, hidden_units]))(din_attention)

    # keras.layers.BatchNormalization实现暂时有坑，借用paddle相关代码实现
    din_attention_fc = keras.layers.Dense(63802)(din_attention)  # (batch_size, item_count + cate_count)
    # item_count:  63001   cate_count:  801         hidden_units:  128   (batch_size, item_count + cate_count + hidden_units)
    din_item = keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([i_emb, din_attention_fc])
    din_item = share_weights()(din_item)  # (batch_size, 1)

    print("logits:", din_item, target_item_bias_emb)
    logits = keras.layers.Add()([din_item, target_item_bias_emb])

    label_model = keras.models.Model(inputs=[hist_item_seq, hist_cate_seq, target_item, target_cate, hist_len], outputs=[logits])

    train_model = keras.models.Model(inputs=[hist_item_seq, hist_cate_seq, target_item, target_cate, hist_len, label],
                               outputs=logits)

    # 计算损失函数
    loss = K.binary_crossentropy(target=label, output=logits, from_logits=True)
    train_model.add_loss(loss)
    train_model.compile(optimizer=keras.optimizers.SGD(1e-3), metrics=["accuracy"])

    return train_model, label_model