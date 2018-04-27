"""Convert individual layers."""
import keras
import numpy as np

from functools import singledispatch
from typing import Tuple, Callable, List

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as f

from ktp import locally_connected


class SequentialModule(nn.Module):
    """A PyTorch version of a keras Sequential model."""
    def __init__(self, layers: nn.ModuleList, activations: List[Callable]):
        super().__init__()
        self.layers = layers
        self.activations = activations

    def forward(self, input):
        """Each layer gets run, followed by its activation."""
        for layer, act in zip(self.layers, self.activations):
            input = act(layer(input))
        return input


def translate_sequential_model(model: keras.Sequential) -> nn.Module:
    """Translate a sequential model."""
    modules = nn.ModuleList()
    activations = []
    for layer in model.layers:
        translated, activation = translate_layer(layer)
        modules.append(translated)
        activations.append(activations)


@singledispatch
def translate_layer(layer: keras.layers.Layer) -> Tuple[nn.Module, Callable]:
    """Translate a single layer."""
    raise NotImplementedError


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


def translate_fully_connected(layer: keras.layers.Dense) -> Tuple[nn.Module, Callable]:
    """Translate a fully connected layer."""
    _, input_size = layer.input_shape
    _, output_size = layer.output_shape
    kernel_weights, bias_weights = layer.get_weights()

    pt_dense = nn.Linear(input_size, output_size, bias=layer.use_bias)
    pt_dense.weight = nn.Parameter(torch.FloatTensor(kernel_weights.reshape(tuple(pt_dense.weight.shape))))
    pt_dense.bias = nn.Parameter(torch.FloatTensor(bias_weights.reshape(tuple(pt_dense.bias.shape))))

    activation = translate_activation(layer.activation)

    return pt_dense, activation


def translate_1d_locally_connected(layer: keras.layers.LocallyConnected1D) -> Tuple[nn.Module, Callable]:
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
    pt_local_conv.weight = nn.Parameter(permuted)

    reshaped_bias = bias_weights.reshape(pt_local_conv.bias.shape)
    pt_local_conv.bias = Parameter(torch.FloatTensor(reshaped_bias))

    activation = translate_activation(layer.activation)

    return pt_local_conv, activation


# pt = locally_connected.Conv2dLocal(
#     in_height=input_height,
#     in_width=input_width,
#     in_channels=in_channels,
#     out_channels=filters,
#     kernel_size=(kernel_height, kernel_width),
#     stride=(stride_y, stride_x),
#     bias=False
# ).cuda()
#
# k_weights = model.get_weights()[0]

# k_shape = (pt.out_height, pt.out_width, kernel_height, kernel_width, in_channels, filters)
# reshaped = k_weights.reshape(k_shape)
# as_tensor = torch.FloatTensor(reshaped)
# permuted = as_tensor.permute((0, 1, 5, 4, 2, 3))
# shape = (pt.out_height, pt.out_width, pt.out_channels, in_channels, kernel_height, kernel_width)
# # assert pt.out_height == keras_local.output_shape[2]
# # assert pt.out_width == keras_local.output_shape[3]
# # assert pt.out_channels == keras_local.output_shape[1] == filters
# # ipdb.set_trace()
# pt.weight = torch.nn.Parameter(permuted.cuda())

def translate_2d_locally_connected(layer: keras.layers.LocallyConnected2D) -> Tuple[nn.Module, Callable]:
    """Translate a 2-dimensional locally connected layer."""
    # Extract various size and shape parameters
    if layer.data_format == "channels_first":
        in_channels, input_width, input_height = layer.input.shape.as_list()[-3:]
    elif layer.data_format == "channels_last":
        input_width, input_height, in_channels = layer.input.shape.as_list()[-3:]
    _, _, filters = layer.output.shape.as_list()[-3:]
    kernel_height, kernel_width = layer.kernel_size
    stride_height, stride_width = layer.strides

    keras_weights = layer.get_weights()
    if len(keras_weights) == 1:
        kernel_weights = keras_weights[0]
        bias_weights = None
        assert layer.use_bias is False
    elif len(keras_weights) == 2:
        kernel_weights, bias_weights = keras_weights
    else:
        raise NotImplementedError("Expected keras_weights to be of length 1 or 2. Was: {}. Full weights: {}".format(len(keras_weights), keras_weights))

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
    pt_local_conv.weight = nn.Parameter(permuted)

    if bias_weights is not None:
        reshaped_bias = bias_weights.reshape(pt_local_conv.bias.shape)
        pt_local_conv.bias = Parameter(torch.FloatTensor(reshaped_bias))

    activation = translate_activation(layer.activation)

    return pt_local_conv, activation


def translate_flatten(layer: keras.layers.Flatten) -> Tuple[nn.Module, Callable]:
    """Translate a PyTorch flatten layer."""
    raise NotImplementedError


def translate_conv1d(layer: keras.layers.Conv1D) -> Tuple[nn.Module, Callable]:
    """Translate a 1D convolution"""
    raise NotImplementedError


def translate_conv2d(layer: keras.layers.Conv2D) -> Tuple[nn.Module, Callable]:
    """Translate a 2D convolution"""
    raise NotImplementedError
