#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import mxnet as mx
import numpy as np
import pickle
import os


def read_images_data(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    with open(file_path, 'rb') as f:
        data_set = pickle.load(f, encoding='bytes')
    images = np.array(data_set[b'data'])
    labels = np.array(data_set[b'labels'])
    return images, labels


def read_train_data(data_path, num_example=50000, batch_size=10000, n_channels=3, size=32):
    i = 0
    images_set = np.zeros(shape=(num_example, n_channels*size*size))
    labels_set = np.zeros(shape=num_example)
    for file in ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"):
        images, labels = read_images_data(data_path, file)
        images_set[batch_size*i: batch_size*(i+1), :] = images
        labels_set[batch_size*i: batch_size*(i+1)] = labels
        i += 1
    images_set = np.reshape(images_set, (num_example, n_channels, size, size))
    return images_set, labels_set


def read_test_data(data_path, num_example=10000, n_channels=3, size=32):
    images_set, labels_set = read_images_data(data_path, 'test_batch')
    images_set = np.reshape(images_set, (num_example, n_channels, size, size))
    return images_set, labels_set


def get_data_iter(data_path, batch_size=64):
    train_data, train_labels = read_train_data(data_path=data_path)
    test_data, test_label = read_test_data(data_path=data_path)
    train_iter = mx.io.NDArrayIter(train_data, train_labels, batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)
    return train_iter, test_iter




