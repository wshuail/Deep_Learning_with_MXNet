#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu')
    )
    if max_pooling:
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
    return net


blk = mlpconv(64, 3, 0)
blk.initialize()

x = nd.random.uniform(shape=(32, 3, 16, 16))  # (16, 16) => (14, 14) => (6, 6) (14-3)/2+1
# y = blk(x)
# print(y.shape)


net = nn.Sequential()
with net.name_scope():
    net.add(
        mlpconv(channels=96, kernel_size=11, padding=0),
        mlpconv(channels=256, kernel_size=5, padding=2),
        mlpconv(channels=384, kernel_size=3, padding=1),
        nn.Dropout(0.5),
        mlpconv(channels=10, kernel_size=3, padding=1, max_pooling=False),
        nn.AvgPool2D(pool_size=5),
        nn.Flatten()
    )
ctx = get_ctx()
net.initialize(ctx=ctx, init=init.Xavier())

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size=batch_size,
                                                transform=transform_resize)

softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.1})

train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=1)






