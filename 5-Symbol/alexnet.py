#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import mxnet as mx


def get_symbol(num_classes):
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(11, 11), stride=4,
                                  num_filter=96)
    act1 = mx.symbol.Activation(data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=act1, pool_type='max', kernel=(3, 3),
                              stride=(2, 2))

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), pad=(2, 2),
                                  num_filter=256)
    act2 = mx.symbol.Activation(data=conv2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=act2, pool_type='max', kernel=(3, 3),
                              stride=(2, 2))

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1),
                                  num_filter=384)
    act3 = mx.symbol.Activation(data=conv3, act_type='relu')
    conv4 = mx.symbol.Convolution(data=act3, kernel=(3, 3), pad=(1, 1),
                                  num_filter=384)
    act4 = mx.symbol.Activation(data=conv4, act_type='relu')
    conv5 = mx.symbol.Convolution(data=act4, kernel=(3, 3), pad=(1, 1),
                                  num_filter=256)
    act5 = mx.symbol.Activation(data=conv5, act_type='relu')
    pool3 = mx.symbol.Pooling(data=act5, pool_type='max', kernel=(3, 3),
                              stride=(2, 2))

    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name='fc1')
    act6 = mx.symbol.Activation(data=fc1, act_type='relu')
    dropout1 = mx.symbol.Dropout(data=act6, p=0.5)

    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096, name='fc2')
    act7 = mx.symbol.Activation(data=fc2, act_type='relu')
    dropout2 = mx.symbol.Dropout(data=act7, p=0.5)

    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
    output = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return output



