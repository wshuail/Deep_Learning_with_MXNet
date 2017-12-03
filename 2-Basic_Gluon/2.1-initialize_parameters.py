#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet.gluon import nn
from mxnet import init


x = nd.random_uniform(shape=(3, 4))


def build_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation='relu'))
        net.add(nn.Dense(2))
    return net


net = build_net()
net.initialize()
y = net(x)
# print(y)


w, b = net[0].weight, net[0].bias
# print('net name %s weight %s bias %s' %(net[0].name, w, b))


# print('weight: ', w.data())
# print('weight gradient: ', w.grad())
# print('bias: ', b.data())
# print('bias gradient: ', b.grad())


params = net.collect_params()
# for param in params:
#     print('param: ', param)


# Different Initialization
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
# print(net[0].weight.data(), net[0].bias.data())

params.initialize(init=init.One(), force_reinit=True)
# print(net[0].weight.data(), net[0].bias.data())


net = build_net()
params = net.collect_params()
# print(params)

net.initialize()
params2 = net.collect_params()
# print(params2)

net(x)
params3 = net.collect_params()
# print(params3)


# Share Parameters
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu'))
    net.add(nn.Dense(4, activation='relu', params=net[-1].params))
    net.add(nn.Dense(2))
net.initialize()
net(x)
# print('second layer params:\n ', net[1].weight.data())
# print('third layer params:\n ', net[2].weight.data())


# Define Init Methods
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True

    def _init_weight(self, _, arr):
        print('init weight: ', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)


net = build_net()
net.initialize(MyInit())
net(x)
# print('Weight: ', net[0].weight.data())


# Set Weight
net = build_net()
net.initialize()
net(x)

w = net[1].weight
w.set_data(nd.ones(shape=w.shape))
print('Weight: ', net[1].weight.data())




