#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2

        self.conv_1 = nn.Conv2D(channels=channels, kernel_size=3, padding=1,
                                strides=strides)
        self.bn_1 = nn.BatchNorm()
        self.conv_2 = nn.Conv2D(channels=channels, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm()

        if not same_shape:
            self.conv_3 = nn.Conv2D(channels=channels, kernel_size=1, strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn_1(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))
        if not self.same_shape:
            x = self.conv_3(x)
        return nd.relu(out + x)


# blk = Residual(3)
# blk.initialize()
# x = nd.random.uniform(shape=(4, 3, 6, 6))
# print(blk(x).shape)
#
# blk2 = Residual(3, same_shape=False)
# blk2.initialize()
# x = nd.random.uniform(shape=(4, 3, 6, 6))
# print(blk2(x).shape)


class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(channels=64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' %(i+1, out.shape))
        return out


net = ResNet(10, verbose=True)
net.initialize()
x = nd.random.uniform(shape=(4, 3, 96, 96))
net(x)


ctx = get_ctx()
net = ResNet(10, verbose=False)
net.initialize(ctx=ctx, init=init.Xavier())

batch_size = 64
train_data, test_data = load_data_fashion_mnist(batch_size, transform_resize)

softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.02})
train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=1)








