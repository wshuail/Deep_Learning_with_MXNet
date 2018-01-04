#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


def get_symbol(num_classes=10):
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128, name='fc1')
    act1 = mx.sym.Activation(data=fc1, act_type='relu', name='act1')
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, name='fc2')
    act2 = mx.sym.Activation(data=fc2, act_type='relu', name='act2')
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=num_classes, name='fc3')
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return mlp


mnist = mx.test_utils.get_mnist()
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


mlp = get_symbol(num_classes=10)

# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes


test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)

test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96




