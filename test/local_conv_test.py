"""Comparison tests between different implementations of locally connected layers."""
import numpy as np
import keras
import torch
from torch.autograd import Variable

from ktp import locally_connected

import ipdb
import pytest


def test_first():
    a = locally_connected.Conv2dLocal(in_channels=256,  out_channels=256, in_height=8, in_width=8, kernel_size=3, stride=1, padding=0)
    input = Variable(torch.randn(1, 256, 8, 8))
    output = a(input)


def test_dense_network():
    pass


def test_flattened_keras():
    input_height = 32
    input_width = 1
    in_channels = 10
    kernel_size = 3
    kernel_width = 1
    filters = 9
    stride_x = 1
    stride_y = 1
    input1_shape = (input_height, in_channels)
    input2_shape = (input_height, 1, in_channels)
    model1 = keras.Sequential()
    model1.add(keras.layers.LocallyConnected1D(filters, kernel_size, input_shape=input1_shape))
    model2 = keras.Sequential()
    model2.add(keras.layers.LocallyConnected2D(filters, (kernel_size, 1), input_shape=input2_shape))
    x = np.zeros((1, input_height, in_channels), dtype=np.float32)
    x2 = x.reshape([1, input_height, 1, in_channels])
    first_pred = model1.predict(x)
    second_pred = model2.predict(x2)
    ipdb.set_trace()
    # var = Variable(torch.from_numpy(x)).cuda()
    # k_res = model.predict(x)


def test_flattened_2d_local():
    input_height = 32
    output_height = 30
    input_width = 1
    in_channels = 10
    kernel_size = 3
    kernel_width = 1
    filters = 9
    stride_x = 1
    stride_y = 1
    input1_shape = (input_height, in_channels)
    input2_shape = (input_height, 1, in_channels)

    model = keras.Sequential()
    model.add(keras.layers.LocallyConnected1D(filters, kernel_size, input_shape=input1_shape))
    pt = locally_connected.Conv2dLocal(
        in_height=input_height,
        in_width=1,
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=(kernel_size, 1),
        stride=(stride_y, 1),
        bias=False
    ).cuda()

    x = np.zeros(input1_shape, dtype=np.float32)
    input1 = x.reshape((1, input_height, in_channels))
    input2 = x.transpose().reshape((1, in_channels, input_height, 1))
    var = Variable(torch.from_numpy(input2)).cuda()
    k_res = model.predict(input1).reshape((output_height, filters))
    t_res = pt(var).cpu().data.numpy().reshape((filters, output_height)).transpose()
    ipdb.set_trace()
    assert (k_res == t_res).all()


# TODO: Copy weights between models.
@pytest.mark.skip()
def test_compare_2d_local():
    for i in range(1):
        input_height = np.random.randint(5, 64)
        input_width = np.random.randint(5, 64)
        in_channels = np.random.randint(1, 8)
        kernel_height = 3
        kernel_width = 2
        filters = np.random.randint(5, 64)
        stride_x = 1
        stride_y = 1
        model = keras.Sequential()
        model.add(keras.layers.LocallyConnected2D(filters, (kernel_height, kernel_width), use_bias=False, input_shape=(in_channels, input_height, input_width),
                                                  data_format="channels_first"))
        pt = locally_connected.Conv2dLocal(
            in_height=input_height,
            in_width=input_width,
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=(kernel_height, kernel_width),
            stride=(stride_y, stride_x),
            bias=False
        ).cuda()

        x = np.zeros((4, in_channels, input_height, input_width), dtype=np.float32)
        var = Variable(torch.from_numpy(x)).cuda()
        k_res = model.predict(x)
        t_res = pt(var).cpu().data.numpy()
        assert (k_res == t_res).all()


def test_compare_flattened_2d_and_1d():
    pass
