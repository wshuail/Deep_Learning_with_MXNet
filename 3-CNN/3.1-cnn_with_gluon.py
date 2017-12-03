#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from utils import load_data_fashion_mnist
from utils import train
from utils import get_ctx

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation='relu'),
        nn.Dense(10)
    )


ctx = get_ctx()
net.initialize(ctx=ctx)


batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


loss_op = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.3})
train(train_data=train_data, test_data=test_data, net=net, loss_op=loss_op,
      trainer=trainer, batch_size=batch_size, epochs=5)

