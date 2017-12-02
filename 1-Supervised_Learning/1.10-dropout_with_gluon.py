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

dropout_prob_1 = 0.2
dropout_prob_2 = 0.5

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_hidden_1, activation='relu'))
    net.add(gluon.nn.Dropout(dropout_prob_1))
    net.add(gluon.nn.Dense(num_hidden_2, activation='relu'))
    net.add(gluon.nn.Dropout(dropout_prob_2))
    net.add(gluon.nn.Dense(num_outputs))
net.initialize()

learning_rate = 0.5
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': learning_rate})
softmax_xentropy = gluon.loss.SoftmaxCrossEntropyLoss()



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
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print('Epoch %s Loss %s Train Accuracy %s Test Accuracy %s' %
          (epoch, train_loss, train_acc/len(train_data), test_acc))

