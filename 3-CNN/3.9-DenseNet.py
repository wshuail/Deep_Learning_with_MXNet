#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


def conv_block(channels):
    net = nn.Sequential()
    net.add(
        nn.BatchNorm(),
        nn.Activation(activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=3, padding=1)
    )
    return net


class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x


# dblk = DenseBlock(2, 10)
# dblk.initialize()
# x = nd.random.uniform(shape=(4, 3, 8, 8))
# print(dblk(x).shape)


def transition_block(channels):
    net = nn.Sequential()
    net.add(
        nn.BatchNorm(),
        nn.Activation(activation='relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return net


# tblk = transition_block(10)
# tblk.initialize()
# x = nd.random.uniform(shape=(4, 3, 8, 8))
# print(tblk(x).shape)


init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10


def dense_net():
    net = nn.Sequential()
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # second block
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers*growth_rate
            if i != len(block_layers) - 1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net


ctx = get_ctx()
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())

batch_size = 64
train_data, test_data = load_data_fashion_mnist(batch_size, transform_resize)

softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.02})
train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=1)

