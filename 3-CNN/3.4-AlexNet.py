#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train


def transform(data, label, resize=224):
    data = mx.image.imresize(data, resize, resize)
    data = mx.nd.transpose(data, (2, 0, 1))
    return data.astype('float32')/255, label.astype('float32')


def load_data_fashion_mnist(batch_size):
    # Create Data
    data_root = '../data/fashion_mnist'
    mnist_train = gluon.data.vision.FashionMNIST(root=data_root, train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(root=data_root, train=False, transform=transform)
    # Read Data
    train_data = gluon.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)
    return train_data, test_data



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
train_data, test_data = load_data_fashion_mnist(batch_size=batch_size)


ctx = mx.gpu()
net.initialize(ctx=ctx, init=init.Xavier())
loss_op = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.01})
train(train_data, test_data, net, loss_op, trainer, batch_size=batch_size, epochs=1)

