#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import random


num_train = 20
num_test = 100
num_inputs = 200


# Get Dataset
true_w = nd.ones(shape=(num_inputs, 1)) * 0.01
true_b = 0.05

X = nd.random_normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w)
y += 0.01*nd.random.normal(shape=y.shape)

x_train, x_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]


batch_size = 1


def data_iter(number_examples):
    idx = list(range(number_examples))
    random.shuffle(idx)
    for i in range(0, number_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, number_examples)])
        yield X.take(j), y.take(j)


# Initialize params
def get_params():
    w = nd.random.normal(shape=(num_inputs, 1))*0.1
    b = nd.zeros((1, ))
    for param in (w, b):
        param.attach_grad()
    return (w, b)


# Regularization
def L2_regularization(w, b):
    return (w**2).sum() + b**2


def net(X, w, b):
    return nd.dot(X, w) + b


def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def test(params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()


def train(lambd):
    epochs = 10
    learning_rate = 0.002
    params = get_params()
    train_loss = []
    test_loss = []

    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, *params)
                loss = square_loss(output, label) + lambd*L2_regularization(*params)
            loss.backward()
            SGD(params, learning_rate)
        train_loss.append(test(params, x_train, y_train))
        test_loss.append(test(params, x_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    train(2)
