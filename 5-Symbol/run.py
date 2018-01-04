#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
import mxnet as mx
from utils.load_cifar10 import get_data_iter
import lenet


batch_size = 64
data_path = "../data/cifar-10-batches-py/"
train_iter, test_iter = get_data_iter(data_path=data_path, batch_size=batch_size)

net = lenet.get_symbol(num_classes=10)

# create a trainable module on CPU
model = mx.mod.Module(symbol=net, context=mx.cpu())
model.fit(train_iter,  # train data
          eval_data=test_iter,  # validation data
          optimizer='sgd',  # use SGD to train
          optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
          eval_metric='acc',  # report accuracy during training
          batch_end_callback=mx.callback.Speedometer(batch_size, 100),  # output progress for each 100 data batches
          num_epoch=10)  # train for at most 10 dataset passes

acc = mx.metric.Accuracy()
model.score(test_iter, acc)
print(acc)