#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import load_data_fashion_mnist
from utils import accuracy, evaluate_accuracy


# Create and Read Data
batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


# Initialize Params
num_inputs = 28*28
num_outputs = 10
num_hidden = 256
weight_scale = 0.01

w1 = nd.random_normal(shape=[num_inputs, num_hidden])
b1 = nd.random_normal(shape=(num_hidden))

w2 = nd.random_normal(shape=[num_hidden, num_outputs])
b2 = nd.random_normal(shape=(num_outputs))

params = [w1, b1, w2, b2]

for param in params:
    param.attach_grad()


# Activation Function
def relu(X):
    return nd.maximum(X, 0)


# Define Model
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, w1) + b1)
    y = nd.dot(h1, w2) + b2
    return y


# Loss
softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()  # more stable


# Optimization
def SGD(params, lr):
    for param in params:
        param[:] = param - lr*param.grad


# Train
epochs = 5
learning_rate = .1

for e in range(epochs):
    total_loss = 0.
    total_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_xentropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)
        total_loss += nd.mean(loss).asscalar()
        acc = accuracy(output, label)
        total_acc += acc
    test_acc = evaluate_accuracy(test_data, net)
    print('Episode %d Loss %s Train Accuracy %s Test Accuracy %s' %
          (e, total_loss/len(train_data), total_acc/len(train_data), test_acc))

