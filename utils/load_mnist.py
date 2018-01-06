#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import struct
import gzip
import mxnet as mx
import numpy as np


def load_gzip_file(data_path):
    with gzip.open(data_path, 'rb') as f:
        f_content = f.read()
    return f_content


def get_image(bytes_buffer):
    image_index = 0
    images = []
    _, num_items, n_rows, n_cols = struct.unpack_from('>IIII', bytes_buffer, image_index)
    image_index += struct.calcsize('IIII')

    for i in range(num_items):
        temp = struct.unpack_from('>784B', bytes_buffer, image_index)
        images.append(temp)
        image_index += struct.calcsize('>784B')
    images = np.array(images).reshape((num_items, 1, n_rows, n_cols))
    return images


def get_label(bytes_buffer):
    index = 0
    labels = []
    _, num_items = struct.unpack_from('>II', bytes_buffer, index)
    index += struct.calcsize('>II')
    for i in range(num_items):
        label = struct.unpack_from('>1B', bytes_buffer, index)
        labels.append(label)
        index += struct.calcsize('>1B')
    labels = np.array(labels).reshape((num_items, ))
    return labels


def load_mnist(data_path, batch_size=32):
    train_image_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
    train_image_bytes = load_gzip_file(data_path=train_image_path)
    train_image = get_image(train_image_bytes)
    train_label_path = os.path.join(data_path, 'train-labels-idx1-ubyte.gz')
    train_label_bytes = load_gzip_file(data_path=train_label_path)
    train_label = get_label(train_label_bytes)
    train_iter = mx.io.NDArrayIter(data=train_image, label=train_label, shuffle=True,
                                   batch_size=batch_size)

    test_image_path = os.path.join(data_path, 't10k-images-idx3-ubyte.gz')
    test_image_bytes = load_gzip_file(data_path=test_image_path)
    test_image = get_image(test_image_bytes)
    test_label_path = os.path.join(data_path, 't10k-labels-idx1-ubyte.gz')
    test_label_bytes = load_gzip_file(data_path=test_label_path)
    test_label = get_label(test_label_bytes)
    test_iter = mx.io.NDArrayIter(data=test_image, label=test_label, shuffle=True,
                                  batch_size=batch_size)

    return train_iter, test_iter





