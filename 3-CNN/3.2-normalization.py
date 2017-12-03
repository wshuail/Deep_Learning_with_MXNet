#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from utils import load_data_fashion_mnist
from utils import accuracy, evaluate_accuracy
from utils import SGD


# Simple Normalization
def pure_batch_norm(x, gamma, beta, eps=1e-5):
    assert len(x.shape) in (2, 4)
    if len(x.shape) == 2:
        mean = x.mean(axis=0)
        variance = ((x-mean)**2).mean(axis=0)
    else:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((x-mean)**2).mean(axis=(0, 2, 3), keepdims=True)
    x_hat = (x - mean)/nd.sqrt(variance + eps)
    return gamma.reshape(mean.shape)*x_hat + beta.reshape(mean.shape)


A = nd.arange(6).reshape((3, 2))
# print(A)
A_norm = pure_batch_norm(A, gamma=nd.array([1, 1]), beta=nd.array([0, 0]))
# print(A_norm)

B = nd.arange(18).reshape((1, 2, 3, 3))
# print(B)
B_norm = pure_batch_norm(B, gamma=nd.array([1, 1]), beta=nd.array([0, 0]))
# print(B_norm)


# Batch Normalization
def batch_norm(x, gamma, beta, is_training, moving_mean, moving_variance,
               eps=1e-5, moving_momentum=0.9):
    assert len(x.shape) in (2, 4)
    if len(x) == 2:
        mean = x.mean(axis=0)
        variance = ((x-mean)**2).mean(axis=0)
    else:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((x-mean)**2).mean(axis=(0, 2, 3), keepdims=True)

        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(variance.shape)

    if is_training:
        x_hat = (x-mean)/nd.sqrt(variance+eps)
        moving_mean[:] = moving_momentum*moving_mean + (1-moving_momentum)*mean
        moving_variance[:] = moving_momentum*moving_variance + (1-moving_momentum)*variance
    else:
        x_hat = (x - moving_mean)/nd.sqrt(moving_variance+eps)

    return gamma.reshape(mean.shape)*x_hat + beta.reshape(mean.shape)


ctx = mx.cpu()


weight_scale = 0.01
c1 = 20
w1 = nd.random.normal(shape=(c1, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros((c1))

gamma_1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta_1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)

moving_mean_1 = nd.zeros(shape=c1, ctx=ctx)
moving_variance_1 = nd.zeros(shape=c1, ctx=ctx)

c2 = 50
w2 = nd.random.normal(shape=(c2, c1, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros((c2))

gamma_2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta_2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean_2 = nd.zeros(shape=c2, ctx=ctx)
moving_variance_2 = nd.zeros(shape=c2, ctx=ctx)

o3 = 128
w3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
b3 = nd.random.normal(shape=o3, scale=weight_scale, ctx=ctx)

w4 = nd.random.normal(shape=(w3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.random.normal(shape=w4.shape[1], ctx=ctx)

params = [w1, b1, w2, b2, w3, b3, w4, b4]
for param in params:
    param.attach_grad()


def net(x, is_training=False):
    x = x.as_in_context(w1.context)
    # print('x shape: ', x.shape)
    h1_conv = nd.Convolution(data=x, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=c1)
    h1_bn = batch_norm(h1_conv, gamma_1, beta_1, is_training, moving_mean_1, moving_variance_1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))

    h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=c2)
    h2_bn = batch_norm(h2_conv, gamma_2, beta_2, is_training, moving_mean_2, moving_variance_2)
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(h2_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
    h2 = h2.flatten()

    h3_linear = nd.dot(h2, w3) + b3
    h3 = nd.relu(h3_linear)

    h4_linear = nd.dot(h3, w4) + b4

    return h4_linear


batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size=batch_size)
softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
leanring_rate = 0.2
epochs = 5


for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_xentropy(output, label)
        loss.backward()
        SGD(params, leanring_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print('epoch %s train accuracy %s test accuracy %s' % (epoch, train_acc / len(train_data), test_acc))




