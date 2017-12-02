#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
import random


# Create Data
num_inputs = 2
num_examples = 1000
batch_size = 10

true_w = [2, -3.4]
true_b = 4.2

X = nd.normal(shape=[num_examples, num_inputs])
y = true_w[0]*X[:, 0] + true_w[1]*X[:, 1] + true_b
y += 0.01*nd.random_normal(shape=y.shape)

# plt.scatter(X[:, 0].asnumpy(), y.asnumpy())
# plt.show()


# Read Data
def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


# Initialize Params
w = nd.random_normal(shape=[num_inputs, 1])
b = nd.zeros((1, ))
params = [w, b]

for param in params:
    param.attach_grad()


# Define Model
def net(X):
    return nd.dot(X, w) + b


# Loss
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape))**2


# Optimization
def SGD(params, lr):
    for param in params:
        param[:] = param - lr*param.grad


# Train
epochs = 5
learning_rate = .001

for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()
    avg_loss = total_loss / num_examples
    print('Episode %d Avg Loss/Example %.6f' % (e, avg_loss))

