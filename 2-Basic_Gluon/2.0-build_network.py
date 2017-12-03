#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet.gluon import nn


# nn.Sequential
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256))
    net.add(nn.Dense(10))
net.initialize()


# nn.Block
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))


net2 = MLP()
# print(net2)
net2.initialize()
x = nd.random.uniform(shape=(4, 20))
y2 = net2(x)
# print(y2)


# print('default name: ', net2.dense0.name)
net3 = MLP(prefix='net3')
# print('net3 name: ', net3.dense0.name)


# Sequential again
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)

    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x


net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation='relu'))
    net4.add(nn.Dense(10))
net4.initialize()
y4 = net4(x)
# print(y4)


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256, 20))

    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x


fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y_fancy_mlp = fancy_mlp(x)
# print(y_fancy_mlp.shape)


class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation='relu'))
            self.net.add(nn.Dense(128, activation='relu'))
            self.dense = nn.Dense(64)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))


rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
# print(rec_mlp)

