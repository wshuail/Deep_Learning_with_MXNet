#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from utils import train
from utils import load_data_fashion_mnist, transform_resize
from utils import get_ctx


class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path 1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
        # path 2
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                   activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                   activation='relu')

        # path 4
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)



icp = Inception(64, 96, 128, 16, 32, 32)
icp.initialize()
# x = nd.random.uniform(shape=(32, 3, 64, 64))
# y = icp(x)
# print(y.shape)


class GoogleNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogleNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(channels=64, kernel_size=7, strides=2,
                          padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(channels=64, kernel_size=1),
                nn.Conv2D(channels=192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 3
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 4
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 5
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )

            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.Flatten(),
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


net = GoogleNet(10, verbose=True)
net.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
y = net(x)

ctx = get_ctx()
net = GoogleNet(10, verbose=False)
net.initialize(ctx=ctx, init=init.Xavier())


batch_size = 64
train_data, test_data = load_data_fashion_mnist(batch_size=batch_size,
                                                transform=transform_resize)
softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.01})

train(train_data, test_data, net, softmax_xentropy, trainer, batch_size, epochs=1)

