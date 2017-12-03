#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from utils import load_data_fashion_mnist
from utils import accuracy, evaluate_accuracy
from utils import get_ctx


# Convolutional Layer
# input data shape (batch_size, channel, height, width)
# weight shape (output_channel, input_channel, height, width)
# bias shape (output_channel)
w = nd.arange(4).reshape((1, 1, 2, 2))
kernel = w.shape[2:]
num_filter = w.shape[1]
b = nd.array([1])
data = nd.arange(9).reshape((1, 1, 3, 3))
out = nd.Convolution(data=data, weight=w, bias=b, kernel=kernel, num_filter=num_filter)


# stride and padding
out2 = nd.Convolution(data=data, weight=w, bias=b, kernel=kernel, num_filter=num_filter,
                      stride=(2, 2), pad=(1, 1))

# Input Multiple Channels
data = nd.arange(18).reshape((1, 2, 3, 3))
w = nd.arange(8).reshape((1, 2, 2, 2))
kernel = w.shape[2:]
num_filter = w.shape[0]
b = nd.array([1])
out3 = nd.Convolution(data=data, weight=w, bias=b, kernel=kernel, num_filter=num_filter)
# print('input data: ', data)
# print('weight: ', w)
# print('bias: ', b)
# print('out: ', out3)


# Output Multiple Channels
data = nd.arange(18).reshape((1, 2, 3, 3))
w = nd.arange(16).reshape((2, 2, 2, 2))
kernel = w.shape[2:]
num_filter = w.shape[0]
b = nd.array([1, 2])
out4 = nd.Convolution(data=data, weight=w, bias=b, kernel=kernel, num_filter=num_filter)
# print('input data: ', data)
# print('weight: ', w)
# print('bias: ', b)
# print('out: ', out4)


# Pooling Layer
data = nd.arange(18).reshape((1, 2, 3, 3))
max_pooling = nd.Pooling(data=data, pool_type='max', kernel=(2, 2))
avg_pooling = nd.Pooling(data=data, pool_type='avg', kernel=(2, 2))
# print('data: ', data)
# print('max_pooling: ', max_pooling)
# print('avg_pooling: ', avg_pooling)


# Build LeNet
batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


ctx = get_ctx()


weight_scale = 0.01
w1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(w1.shape[0], ctx=ctx)

w2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(w2.shape[0], ctx=ctx)

w3 = nd.random_normal(shape=[1250, 128], scale=weight_scale, ctx=ctx)
b3 = nd.zeros(w3.shape[1], ctx=ctx)

w4 = nd.random_normal(shape=[128, 10], scale=weight_scale, ctx=ctx)
b4 = nd.zeros(w4.shape[1], ctx=ctx)


params = [w1, b1, w2, b2, w3, b3, w4, b4]
for param in params:
    param.attach_grad()


def net(x):
    # x = x.as_in_context(w1.context)
    # first conv layer
    # (bs, 1, 28, 28) ==> (bs, 20, 12, 12)
    h1_conv = nd.Convolution(data=x, weight=w1, bias=b1, kernel=w1.shape[2:],
                             num_filter=w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
    # second conv layer
    # (bs, 20, 12, 12) ==> (bs, 50, 5, 5) ==> (bs, 50*5*5)
    h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:],
                             num_filter=w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
    h2 = h2.flatten()
    # first fc layer
    # (bs, 1250) ==> (bs, 128)
    h3 = nd.relu(nd.dot(h2, w3) + b3)
    # second fc layer
    # (bs, 128) ==> (bs, 10)
    h4 = nd.dot(h3, w4) + b4
    return h4


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
epochs = 5
learning_rate = 0.2
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_xentropy(output, label)
        loss.backward()
        SGD(params=params, lr=learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print('epoch %s train accuracy %s test accuracy %s' %(epoch, train_acc/len(train_data), test_acc))


