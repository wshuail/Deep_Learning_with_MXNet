#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import accuracy, evaluate_accuracy


# Create Data

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')


data_root = '../data/fashion_mnist'
mnist_train = gluon.data.vision.FashionMNIST(root=data_root, train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(root=data_root, train=False, transform=transform)


batch_size = 256


# Read Data
train_data = gluon.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)



# Initialize Params
num_inputs = 28*28
num_outputs = 10
w = nd.random_normal(shape=[num_inputs, num_outputs])
b = nd.random_normal(shape=(num_outputs))
params = [w, b]

for param in params:
    param.attach_grad()


# Define Model
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    return exp/partition


def net(X):
    logits = nd.dot(X.reshape((-1, num_inputs)), w) + b
    return softmax(logits)


# Loss
def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)


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
            loss = cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)
        total_loss += nd.mean(loss).asscalar()
        acc = accuracy(output, label)
        total_acc += acc
    test_acc = evaluate_accuracy(test_data, net)
    print('Episode %d Loss %s Train Accuracy %s Test Accuracy %s' %
          (e, total_loss/len(train_data), total_acc/len(train_data), test_acc))

