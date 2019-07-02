# 基于keras的Deep Interest Network实现
欢迎star，后续会有更多的nlp和推荐的代码实现，同时关注我的知乎(https://zhuanlan.zhihu.com/skydm)

参考阿里的论文Deep Interest Network for Click-Through Rate Prediction(https://arxiv.org/abs/1706.06978) 

实现过程参考了了如下代码库：
1. https://github.com/zhougr1993/DeepInterestNetwork
2. https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/din

数据下载：
Amazon Product数据集并进行预处理(http://jmcauley.ucsd.edu/data/amazon/), 这部分的数据处理代码可以参考上述代码库

提供了jupyter notebook的具体实现过程， 调参确实一门学问。
起初在设置如下过程时，发现loss经过几个epoches后，loss上升，同时acc减小。
```
model.compile(optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
Epoch 1/10
40761/40762 [============================>.] - ETA: 0s - loss: 0.5377

Epoch 2/10
40761/40762 [============================>.] - ETA: 0s - loss: 0.5241

Epoch 5/10
40758/40762 [============================>.] - ETA: 0s - loss: 0.5347

Epoch 6/10
40757/40762 [============================>.] - ETA: 0s - loss: 0.5427

Epoch 7/10
40760/40762 [============================>.] - ETA: 0s - loss: 0.5506

Epoch 9/10
40758/40762 [============================>.] - ETA: 0s - loss: 0.5651
```
经过调整优化算法和学习参数之后，loss下降回归正常。

```
Epoch 1/10
40761/40762 [============================>.] - ETA: 0s - loss: 0.6021

Epoch 2/10
40761/40762 [============================>.] - ETA: 0s - loss: 0.5349

Epoch 3/10
40760/40762 [============================>.] - ETA: 0s - loss: 0.5302

Epoch 4/10
40759/40762 [============================>.] - ETA: 0s - loss: 0.5280

Consider using a TensorFlow optimizer from `tf.train`.
acc: 0.7824, best acc: 0.7824
```