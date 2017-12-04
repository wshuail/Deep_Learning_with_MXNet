#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


def vgg_block(num_convs, channels):
    net = nn.Sequential()
    for _ in range(num_convs):
        net.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu')
        )
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net


blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2, 3, 16, 16))
y = blk(x)
print(y.shape)


def vgg_stack(arthitecture):
    net = nn.Sequential()
    for (num_convs, channels) in arthitecture:
        net.add(vgg_block(num_convs, channels))
    return net


num_outputs = 10
architectures = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
net = nn.Sequential()
with net.name_scope():
    net.add(
        vgg_stack(architectures),
        nn.Flatten(),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(num_outputs)
    )
ctx = get_ctx()
net.initialize(init.Xavier())

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size, transform_resize)

softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.05})

train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=1)

