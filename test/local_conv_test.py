"""Comparison tests between different implementations of locally connected layers."""
import numpy as np
import keras
import torch
from torch.autograd import Variable

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb
from tensorflow.python import debug as tf_debug

from ktp import locally_connected, translate

import ipdb
import pytest


def test_first():
    a = locally_connected.Conv2dLocal(in_channels=256, out_channels=256, in_height=8, in_width=8, kernel_size=3, stride=1, padding=0)
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

    sess = keras.backend.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    keras.backend.set_session(sess)

    model = keras.Sequential()
    model.add(keras.layers.LocallyConnected1D(filters, kernel_size, input_shape=input1_shape))
    pt, _ = translate.translate_1d_locally_connected(model.layers[0])
    pt = pt.cuda()
    x = np.arange(input_height * in_channels, dtype=np.float32).reshape(input1_shape)
    input1 = x.reshape((1, input_height, in_channels))
    input2 = x.transpose().reshape((1, in_channels, input_height, 1))
    var = Variable(torch.from_numpy(input2)).cuda()
    k_res = model.predict(input1).reshape((output_height, filters))
    t_res = pt(var).cpu().data.numpy()
    ipdb.set_trace()
    assert (k_res == t_res).all()


# TODO: Copy weights between models.
# @pytest.mark.skip()
def test_compare_2d_local():
    for i in range(1):
        input_height = 4
        input_width = 2
        in_channels = 1
        kernel_height = 3
        kernel_width = 2
        filters = 3
        stride_x = 1
        stride_y = 1
        # sess = keras.backend.get_session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # keras.backend.set_session(sess)
        weights = np.arange(36).reshape((2, 6, 3))

        model = keras.Sequential()
        keras_local = keras.layers.LocallyConnected2D(filters, (kernel_height, kernel_width),
                                                      use_bias=False,
                                                      input_shape=(in_channels, input_height, input_width),
                                                      data_format="channels_first")
        model.add(keras_local)
        model.set_weights([weights])
        # pt = locally_connected.Conv2dLocal(
        #     in_height=input_height,
        #     in_width=input_width,
        #     in_channels=in_channels,
        #     out_channels=filters,
        #     kernel_size=(kernel_height, kernel_width),
        #     stride=(stride_y, stride_x),
        #     bias=False
        # ).cuda()

        # k_weights = model.get_weights()[0]

        # k_shape = (pt.out_height, pt.out_width, kernel_height, kernel_width, in_channels, filters)
        # reshaped = k_weights.reshape(k_shape)
        # as_tensor = torch.FloatTensor(reshaped)
        # permuted = as_tensor.permute((0, 1, 5, 4, 2, 3))
        # shape = (pt.out_height, pt.out_width, pt.out_channels, in_channels, kernel_height, kernel_width)
        # assert pt.out_height == keras_local.output_shape[2]
        # assert pt.out_width == keras_local.output_shape[3]
        # assert pt.out_channels == keras_local.output_shape[1] == filters
        # ipdb.set_trace()
        # pt.weight = torch.nn.Parameter(permuted.cuda())
        # print("Pt init weight is: {}".format(pt.weight))
        # ipdb.set_trace()
        # pt.bias = torch.FloatTensor(k_bias.reshape(tuple(pt.bias)))

        pt = translate.translate_2d_locally_connected(keras_local)[0].cuda()

        x = np.arange(0, 8).reshape((1, in_channels, input_height, input_width)).astype(np.float32)
        var = Variable(torch.from_numpy(x)).cuda()
        k_res = model.predict(x)
        t_res = pt(var).cpu().data.numpy()
        assert (k_res == t_res).all()


#         (0 ,.,.) =
# 0.5417 -0.6260 -0.0848
# -0.0604  0.6027  0.3099
# 0.2878  0.3740 -0.0613
# -0.4248  0.1856 -0.2040
#
# (1 ,.,.) =
# -0.4726  0.5507  0.3702
# -0.2011  0.2885 -0.3012
# -0.3903  0.4651  0.4179
# 0.2640 -0.4609 -0.0684


def test_compare_flattened_2d_and_1d():
    pass
