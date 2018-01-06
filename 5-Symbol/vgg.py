#! /usr/bin/env python
# -*-coding: utf8 -*- 

import mxnet as mx

def build_vgg_block(data, layout, num_filter, batch_norm=False):
    for i, num_layer in enumerate(layout):
        print(num_layer)
        for i in range(num_layer):
            print('i: ', i)
            data = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), pad=(1, 1))
            if batch_norm:
                data = mx.symbol.BatchNorm(data=data)
            data = mx.symbol.Activation(data=data, act_type='relu')
        data = mx.symbol.Pooling(data=data, pool_type='max', kernel=(2, 2), stride=(1, 1))
    return data


def build_classifier(data, num_classes, **kwargs):
    flatten = mx.symbol.Flatten(data=data)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
    act1 = mx.symbol.Activation(data=fc1, act_type='relu')
    drop1 = mx.symbol.Dropout(data=act1, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=512)
    act2 = mx.symbol.Activation(data=fc2, act_type='relu')
    drop2 = mx.symbol.Dropout(data=act2, p=0.5)
    fc3 = mx.symbol.FullyConnected(data=drop2, num_hidden=num_classes)
    output = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return output


def get_symbol(num_classes):
    layout = [1]
    data = mx.symbol.Variable(name='data')
    vgg_block = build_vgg_block(data=data, layout=layout, num_filter=32)
    output = build_classifier(data=data, num_classes=num_classes)
    return output

