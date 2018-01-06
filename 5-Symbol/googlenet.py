#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import mxnet as mx

def conv_act_block(data, num_filter, kernel, stride=(1, 1), pad=(0, 0)):
    data = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    data = mx.symbol.Activation(data=data, act_type='relu')
    return data


def inception_block(data, p1_k1, p2_k1, p2_k2, p3_k1, p3_k2, p4_k1):
    # path 1
    p1 = conv_act_block(data=data, num_filter=p1_k1, kernel=(1, 1))

    # path 2
    p2_b1 = conv_act_block(data=data, num_filter=p2_k1, kernel=(1, 1))
    p2_b2 = conv_act_block(data=p2_b1, num_filter=p2_k2, kernel=(3, 3),
                                      pad=(1, 1))

    # path 3
    p3_b1 = conv_act_block(data=data, num_filter=p3_k1, kernel=(1, 1))
    p3_b2 = conv_act_block(data=p3_b1, num_filter=p3_k2, kernel=(5, 5),
                                      pad=(2, 2))

    # path 4
    p4_p1 = mx.symbol.Pooling(data=data, pool_type='max', kernel=(3, 3),
                                  stride=(1, 1), pad=(1, 1))
    p4_b1 = conv_act_block(data=p4_p1, num_filter=p4_k1, kernel=(1, 1))

    return mx.symbol.concat(*[p1, p2_b2, p3_b2, p4_b1])


def get_symbol(num_classes):
    data = mx.symbol.Variable(name='data')
    incep_1 = inception_block(data, 64, 96, 128, 16, 32, 32)
    flatten = mx.symbol.Flatten(data=incep_1)
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)
    output = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return output


