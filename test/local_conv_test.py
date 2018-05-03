"""Comparison tests between different implementations of locally connected layers.
Honestly I probably should have just defined a commutative diagram and tested everything along that, but oh well.
"""
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


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def test_first():
    a = locally_connected.Conv2dLocal(in_channels=256, out_channels=256, in_height=8, in_width=8, kernel_size=3, stride=1, padding=0)
    input = Variable(torch.randn(1, 256, 8, 8))
    output = a(input)


def test_dense_network():
    pass


def test_compare_flattened_keras():
    """Compare a flattened (i.e. width = 1) 2D locally connected layer to a 1D locally connected layer."""
    for i in range(10):
        input_height = np.random.randint(2, 10)
        in_channels = np.random.randint(1, 10)
        kernel_size = np.random.randint(1, input_height)
        filters = np.random.randint(1, 10)
        input1_shape = (input_height, in_channels)
        input2_shape = (input_height, 1, in_channels)

        # First model
        keras_1d_model = keras.Sequential()
        keras_1d_layer = keras.layers.LocallyConnected1D(filters, kernel_size, input_shape=input1_shape)
        keras_1d_model.add(keras_1d_layer)

        # Flattened keras 2D locally connected
        keras_flattened_2d_model = keras.Sequential()
        keras_flattened_2d_layer = keras.layers.LocallyConnected2D(filters, (kernel_size, 1), input_shape=input2_shape)
        keras_flattened_2d_model.add(keras_flattened_2d_layer)

        # Flattened PyTorch 2D locally connected.
        torch_flattened_model = translate.translate_2d_locally_connected(keras_flattened_2d_layer)[0].cuda()

        # Copy weights
        weights = np.random.uniform(-10, 10, keras_1d_layer.kernel_shape)
        keras_1d_model.set_weights([weights])
        keras_flattened_2d_model.set_weights([weights.reshape(keras_flattened_2d_layer.kernel_shape)])

        # Compare outputs
        keras_1d_input = np.arange(input_height * in_channels).reshape((1, input_height, in_channels))
        keras_2d_input = keras_1d_input.reshape((1, input_height, 1, in_channels))
        keras_input = keras_2d_input.reshape((1, in_channels, input_height, 1)).astype(np.float32)
        torch_input = torch.Tensor(keras_input).cuda()
        first_pred = keras_1d_model.predict(keras_1d_input)
        second_pred = keras_flattened_2d_model.predict(keras_2d_input)
        t_res = torch_flattened_model(torch_input).cpu().data.numpy()
        assert (first_pred.reshape(second_pred.shape) == second_pred == t_res).all()


# TODO: Change to Hypothesis
def test_compare_2d_local_full():
    """Compare keras and PyTorch 2D locally connected layers on randomly generated values."""
    for i in range(50):
        input_height = np.random.randint(2, 15)
        input_width = np.random.randint(2, 15)
        in_channels = np.random.randint(2, 15)
        kernel_height = np.random.randint(1, input_height)
        kernel_width = np.random.randint(1, input_width)
        filters = np.random.randint(1, 20)

        # Create keras model
        keras_model = keras.Sequential()
        keras_local = keras.layers.LocallyConnected2D(filters, (kernel_height, kernel_width),
                                                      use_bias=False,
                                                      input_shape=(in_channels, input_height, input_width),
                                                      data_format="channels_first")
        keras_model.add(keras_local)

        # Setup weights
        weights = np.random.uniform(-10, 10, keras_local.kernel.shape.as_list())
        keras_model.set_weights([weights])

        # Translate keras to PyTorch model
        torch_model = translate.translate_2d_locally_connected(keras_local)[0].cuda()

        keras_input = np.random.uniform(-10, 10, (1, in_channels, input_height, input_width)).astype(np.float32)
        kres = keras_model.predict(keras_input)

        torch_input = torch.Tensor(keras_input.copy()).cuda()
        tres = torch_model(torch_input).cpu().data.numpy()
        assert (kres == tres).all()

        # Cleanup
        keras.backend.clear_session()
        del torch_model


def test_compare_1d_local():
    for i in range(1):
        input_height = np.random.randint(2, 3)
        in_channels = np.random.randint(2, 3)
        kernel_height = np.random.randint(1, input_height)
        filters = 1

        model = keras.Sequential()
        keras_local = keras.layers.LocallyConnected1D(filters, kernel_height, input_shape=(input_height, in_channels), use_bias=False)
        model.add(keras_local)
        weight_size = keras_local.kernel.shape.num_elements()
        weights = np.arange(weight_size).reshape(keras_local.kernel.shape.as_list())
        model.set_weights([weights])

        pt = translate.translate_1d_locally_connected(keras_local)[0].cuda()

        input_size = input_height * in_channels
        x = np.arange(input_size).reshape((1, input_height, in_channels)).astype(np.float32)
        for_torch = torch.Tensor(x.reshape((1, in_channels, input_height, 1))).cuda()
        k_res = model.predict(x)
        t_res = pt(for_torch).cpu().data.numpy()
        assert (k_res == t_res).all()


def test_compare_flattened_2d_and_1d():
    pass
