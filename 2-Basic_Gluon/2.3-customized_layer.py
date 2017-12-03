#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn


# Simple Layer
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - nd.mean(x)


layer = CenteredLayer()
# print(layer(nd.array([1, 2, 3, 4, 5])))


net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
net.initialize()


x = nd.random_uniform(shape=[4, 8])
y = net(x)
# print(nd.mean(y))


# Customized Layer with Params
my_param = gluon.Parameter(name='exciting_params', shape=(3, 3))
my_param.initialize()
# print('weight: ', my_param.data())
# print('gradients: ', my_param.grad())


pd = gluon.ParameterDict(prefix='block1_')
pd.get(name='exciting_params', shape=(3, 3))
# print(pd)


class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__()
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units, ))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


dense = MyDense(units=5, in_units=10, prefix='o_my_dense')
# print(dense.params)
dense.initialize()
# print(dense(nd.random_uniform(shape=(2, 10))))


# No Difference with Gluon Layer
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(units=32, in_units=64))
    net.add(MyDense(units=2, in_units=32))
net.initialize()
# print(net(nd.random_uniform(shape=(2, 64))))

