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


# Define Model and Initialize Params
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation='relu'))
    net.add(gluon.nn.Dense(10))
net.initialize()


# Loss
softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()


# Optimization
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': 0.1})


# Train
epochs = 5
for e in range(epochs):
    total_loss = 0.
    total_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_xentropy(output, label)
        loss.backward()
        trainer.step(batch_size=batch_size)
        total_loss += nd.mean(loss).asscalar()
        acc = accuracy(output, label)
        total_acc += acc
    test_acc = evaluate_accuracy(test_data, net)
    print('Episode %d Loss %s Train Accuracy %s Test Accuracy %s' %
          (e, total_loss/len(train_data), total_acc/len(train_data), test_acc))

