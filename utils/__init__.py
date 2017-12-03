#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd


# def get_ctx():
#     try:
#         ctx = mx.gpu()
#         _ = nd.zeros((1, ), ctx=ctx)
#     except:
#         ctx = mx.cpu()
#     return ctx


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        data = data.reshape((-1, 1, 28, 28))
        output = net(data)
        acc += accuracy(output, label)
    return acc/len(data_iterator)


def transform(data, label, resize=None):
    if resize is not None:
        data = mx.image.imresize(data, resize, resize)
    data = mx.nd.transpose(data, (2, 0, 1))
    return data.astype('float32')/255, label.astype('float32')


def load_data_fashion_mnist(batch_size):
    # Create Data
    data_root = '../data/fashion_mnist'
    mnist_train = gluon.data.vision.FashionMNIST(root=data_root, train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(root=data_root, train=False, transform=transform)
    # Read Data
    train_data = gluon.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)
    return train_data, test_data


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def train(train_data, test_data, net, loss_op, trainer, batch_size, epochs):
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = loss_op(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
        test_acc = evaluate_accuracy(test_data, net)
        print('epoch %s train accuracy %s test accuracy %s' % (epoch, train_acc / len(train_data), test_acc))
