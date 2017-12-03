#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import gluon
from mxnet.gluon import nn
from utils import load_data_fashion_mnist
from utils import train


net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=20, kernel_size=5))
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))

    net.add(nn.Conv2D(channels=50, kernel_size=3))
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Flatten())

    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(10))
net.initialize()


batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.2})
train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=5)
