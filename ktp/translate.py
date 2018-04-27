"""Convert individual layers."""
import keras
import numpy as np

from functools import singledispatch
from typing import Tuple, Callable

import torch
from torch.nn import Parameter
import torch.nn.functional as f

from ktp import locally_connected

import ipdb


@singledispatch
def translate_layer(layer: keras.layers.Layer):
    """Translate a single layer."""
    pass


def translate_activation(activation):
    """Translate a Keras activation function to the corresponding PyTorch activation."""
    act_name = activation.__name__

    if act_name == "tanh":
        return f.tanh
    elif act_name == "sigmoid":
        return f.sigmoid
    elif act_name == "relu":
        return f.relu
    elif act_name == "linear":
        return lambda x: x
    else:
        raise NotImplementedError


def translate_fully_connected(layer: keras.layers.Dense) -> Tuple[torch.nn.Module, Callable]:
    """Translate a fully connected layer."""
    _, input_size = layer.input_shape
    _, output_size = layer.output_shape
    kernel_weights, bias_weights = layer.get_weights()

    pt_dense = torch.nn.Linear(input_size, output_size, bias=layer.use_bias)
    pt_dense.weight = Parameter(torch.FloatTensor(kernel_weights.reshape(tuple(pt_dense.weight.shape))))
    pt_dense.bias = Parameter(torch.FloatTensor(bias_weights.reshape(tuple(pt_dense.bias.shape))))

    activation = translate_activation(layer.activation)

    return pt_dense, activation


def translate_1d_locally_connected(layer: keras.layers.LocallyConnected1D) -> Tuple[torch.nn.Module, Callable]:
    """Translate a 1-dimensional locally connected layer."""
    _, input_height, in_channels = layer.input.shape.as_list()
    _, _, filters = layer.output.shape.as_list()
    kernel_size = layer.kernel_size[0]
    stride = layer.strides[0]
    kernel_weights, bias_weights = layer.get_weights()

    pt_local_conv = locally_connected.Conv2dLocal(
        in_height=input_height,
        in_width=1,
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        bias=layer.use_bias
    )

    k_shape = (pt_local_conv.out_height, pt_local_conv.out_width, kernel_size, 1, in_channels, filters)
    reshaped = kernel_weights.reshape(k_shape)
    as_tensor = torch.FloatTensor(reshaped)
    permuted = as_tensor.permute((0, 1, 5, 4, 2, 3))
    pt_local_conv.weight = torch.nn.Parameter(permuted.cuda())

    reshaped_bias = bias_weights.reshape(pt_local_conv.bias.shape)
    pt_local_conv.bias = Parameter(torch.FloatTensor(reshaped_bias))

    activation = translate_activation(layer.activation)

    return pt_local_conv, activation


def translate_2d_locally_connected(layer: keras.layers.LocallyConnected2D) -> Tuple[torch.nn.Module, Callable]:
    """Translate a 2-dimensional locally connected layer."""
    input_width, input_height, in_channels = layer.input.shape.as_list()
    _, _, filters = layer.output.shape.as_list()
    kernel_height, kernel_width = layer.kernel_size
    stride_height, stride_width = layer.strides
    kernel_weights, bias_weights = layer.get_weights()

    pt_local_conv = locally_connected.Conv2dLocal(
        in_height=input_height,
        in_width=input_width,
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=(kernel_height, kernel_width),
        stride=(stride_height, stride_width),
        bias=layer.use_bias
    )

    k_shape = (pt_local_conv.out_height, pt_local_conv.out_width, kernel_height, kernel_width, in_channels, filters)
    reshaped = kernel_weights.reshape(k_shape)
    as_tensor = torch.FloatTensor(reshaped)
    permuted = as_tensor.permute((0, 1, 5, 4, 2, 3))
    pt_local_conv.weight = torch.nn.Parameter(permuted.cuda())

    reshaped_bias = bias_weights.reshape(pt_local_conv.bias.shape)
    pt_local_conv.bias = Parameter(torch.FloatTensor(reshaped_bias))

    activation = translate_activation(layer.activation)

    return pt_local_conv, activation
