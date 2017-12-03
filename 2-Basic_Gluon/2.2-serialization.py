#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


# Save and Load NDArrays
x = nd.ones(3)
y = nd.zeros(4)
filename = '../data/test1.params'
nd.save(fname=filename, data=[x, y])

a, b = nd.load(filename)
# print('a: ', a)
# print('b: ', b)


# Save and Load Gluon Params
def build_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(2))
    return net


net = build_net()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
print(net(x))

filename = '../data/mlp.params'
net.save_params(filename=filename)


net2 = build_net()
net2.load_params(filename=filename, ctx=mx.cpu())
print(net2(x))

