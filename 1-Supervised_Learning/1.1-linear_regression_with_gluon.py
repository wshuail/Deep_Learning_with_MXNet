#! /usr/bin/env python
# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import random
import matplotlib.pyplot as plt


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
dataset = gluon.data.ArrayDataset(data=X, label=y)
data_iter = gluon.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# Define Model
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))  # no need input size, just output size


# Initialize Params
net.initialize()


# Loss
square_loss = gluon.loss.L2Loss()


# Optimization
trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': 0.1})


epochs = 5
# 训练
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size=batch_size)
        total_loss += nd.sum(loss).asscalar()
    avg_loss = total_loss/num_examples
    print('Episode %d Avg Loss/Example %.6f' %(e, avg_loss))

