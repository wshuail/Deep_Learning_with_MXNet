#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


net = nn.Sequential()
with net.name_scope():
    net.add(
        # 1st stage
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 2rd stage
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 3st stage
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 4th stage
        nn.Flatten(),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        # 5th stage
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        # 6th stage
        nn.Dense(10)
    )


batch_size = 64
train_data, test_data = load_data_fashion_mnist(batch_size=batch_size, transform=transform_resize)


ctx = get_ctx()
net.initialize(ctx=ctx, init=init.Xavier())
loss_op = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.01})
train(train_data, test_data, net, loss_op, trainer, batch_size=batch_size, epochs=1)

