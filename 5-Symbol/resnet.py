#! /usr/bin/env python
#-*- coding: utf-8 -*- 

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import mxnet as mx


def residual_block(data, num_filter, stride, dim_match, bottle_neck, momentum=0.9):
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=2e-5)
        act1 = mx.sym.Activation(data=bn1, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=momentum, eps=2e-5)
        act2 = mx.sym.Activation(data=bn2, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=momentum, eps=2e-5)
        act3 = mx.sym.Activation(data=bn3, act_type='relu')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=2e-5)
        act1 = mx.sym.Activation(data=bn1, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=momentum, eps=2e-5)
        act2 = mx.sym.Activation(data=bn2, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True)

    if dim_match:
        shortcut = data 
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), no_bias=True)
    return conv2 + shortcut

def build_net(num_classes, filters, res_block_units, bottle_neck, momentum=0.9):
    data = mx.sym.Variable('data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=2e-5)
    data = mx.sym.Convolution(data=data, num_filter=filters[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True)
    data = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=2e-5)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    ## Residules Below
    for i in range(len(res_block_units)):
        stride_step = 1 if i == 0 else 2
        data = residual_block(data=data, num_filter=filters[i+1], stride=(stride_step, stride_step), dim_match=False, bottle_neck=bottle_neck)
        for _ in range(res_block_units[i] - 1):
            data = residual_block(data=data, num_filter=filters[i+1], stride=(1, 1), dim_match=True, bottle_neck=bottle_neck)

    ## Residuals Above
    data = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=2e-5)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg')
    data = mx.sym.Flatten(data=data)
    data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)
    output = mx.sym.SoftmaxOutput(data=data, name='softmax')
    return output 


def get_symbol(num_classes, num_layers, image_shape):
    assert len(image_shape) == 3
    n_channels, width, height = image_shape
    if width <= 28:
        num_stages = 3
        if num_layers >= 164 and (num_layers - 2) % 9 == 0:  # bottle_neck=True => Conv_layers=3 => . num_stages => 6
            units = [(num_layers-2)//9]*num_stages
            filters = (16, 64, 128, 256)
            bottle_neck = True
        elif num_layers < 164 and (num_layers - 2) % 6 == 0:
            units = [(num_layers - 2)//6] * num_stages
            filters = (16, 16, 32, 64)
            bottle_neck = False
        else:
            raise ValueError('Invalid number of layers.')
    else:
        if num_layers >= 50:
            filters = (64, 256, 512, 1024, 2048)
            bottle_neck = True
            if num_layers == 50:
                units = (3, 4, 6, 3)
            elif num_layers == 101:
                units = (3, 4, 23, 3)
            elif num_layers == 152:
                units = (3, 8, 36, 3)
            elif num_layers == 200:
                units = (3, 24, 36, 3)
            elif num_layers == 269:
                units = (3, 30, 48, 8)
            else:
                raise ValueError('Invalid number of layers.')
        else:
            filters = (64, 64, 128, 256, 512)
            bottle_neck = False
            if num_layers == 18:
                units = (2, 2, 2, 2)
            elif num_layers == 34:
                units = (3, 4, 6, 3)
            else:
                raise ValueError('Invalid number of layers.')
    return build_net(num_classes=num_classes, filters=filters, res_block_units=units, bottle_neck=bottle_neck)


