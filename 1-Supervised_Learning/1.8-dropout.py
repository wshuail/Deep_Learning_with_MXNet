#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import load_data_fashion_mnist
from utils import accuracy, evaluate_accuracy


def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1
    if keep_probability == 0:
        return X.zeros_like()
    else:
        mask = nd.random.uniform(0, 1, X.shape, ctx=X.context) < keep_probability
        scale = 1/keep_probability
        return mask*X*scale


batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


num_inputs = 28*28
num_outputs = 10

num_hidden_1 = 256
num_hidden_2 = 256
weight_scale = 0.1

w1 = nd.random_normal(shape=[num_inputs, num_hidden_1], scale=weight_scale)
b1 = nd.zeros(num_hidden_1)

w2 = nd.random_normal(shape=[num_hidden_1, num_hidden_2], scale=weight_scale)
b2 = nd.zeros(num_hidden_2)

w3 = nd.random_normal(shape=[num_hidden_2, num_outputs], scale=weight_scale)
b3 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2, w3, b3]

for param in params:
    param.attach_grad()


dropout_prob_1 = 0.2
dropout_prob_2 = 0.3


def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = nd.dot(X, w1) + b1
    h1 = nd.relu(h1)
    h1 = dropout(h1, dropout_prob_1)
    h2 = nd.dot(h1, w2) + b2
    h2 = nd.relu(h2)
    h2 = dropout(h2, dropout_prob_2)
    y = nd.dot(h2, w3) + b3
    return y


softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.5


# Optimization
def SGD(params, lr):
    for param in params:
        param[:] = param - lr*param.grad


epochs = 5
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_xentropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print('Epoch %s Loss %s Train Accuracy %s Test Accuracy %s' %
          (epoch, train_loss, train_acc/len(train_data), test_acc))

