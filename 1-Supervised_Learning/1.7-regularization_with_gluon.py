#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
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
dataset_train = gluon.data.ArrayDataset(data=x_train, label=y_train)
dataset_iter_train = gluon.data.DataLoader(dataset=dataset_train, batch_size=batch_size)


square_loss = gluon.loss.L2Loss()

def data_iter(number_examples):
    idx = list(range(number_examples))
    random.shuffle(idx)
    for i in range(0, number_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, number_examples)])
        yield X.take(j), y.take(j)


net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1))
net.initialize()


def get_trainer(lr, weight_decay):
    trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                            optimizer_params={'learning_rate': lr,
                                              'wd': weight_decay})
    return trainer


def test(net, X, y):
    return square_loss(net(X), y).mean().asscalar()


def train(weight_decay):
    epochs = 10
    learning_rate = 0.005

    trainer = get_trainer(lr=learning_rate, weight_decay=weight_decay)

    train_loss = []
    test_loss = []

    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(test(net, x_train, y_train))
        test_loss.append(test(net, x_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    train(2)
