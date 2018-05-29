"""Convert individual layers."""
import keras
import numpy as np

from functools import singledispatch
from typing import Tuple, Callable, List, Optional

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as f

from ktp import locally_connected


def get_kernel_and_bias(layer) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract kernel and bias weights from a layer that may not be using bias."""
    keras_weights = layer.get_weights()
    if len(keras_weights) == 1:
        kernel_weights = keras_weights[0]
        bias_weights = None
        assert layer.use_bias is False
    elif len(keras_weights) == 2:
        kernel_weights, bias_weights = keras_weights
    else:
        raise NotImplementedError("Expected keras_weights to be of length 1 or 2. Was: {}. Full weights: {}".format(len(keras_weights), keras_weights))
    return kernel_weights, bias_weights


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


def translate_sequential_model(model: keras.models.Sequential) -> nn.Module:
    """Translate a sequential model."""
    modules = nn.ModuleList()
    activations = []
    for layer in model.layers:
        translated, activation = translate_layer(layer)
        modules.append(translated)
        activations.append(activations)
    return modules


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


# TODO: Keras is the transpose of the default Pytorch behavior. Extend this to account for that (transpose the input, transpose the output).
def translate_fully_connected(layer: keras.layers.Dense) -> Tuple[nn.Module, Callable]:
    """Translate a fully connected layer."""
    _, input_size = layer.input_shape
    _, output_size = layer.output_shape
    kernel_weights, bias_weights = get_kernel_and_bias(layer)
    pt_dense = nn.Linear(input_size, output_size, bias=layer.use_bias)
    pt_dense.weight = nn.Parameter(torch.Tensor(kernel_weights.transpose()))

    if bias_weights is not None:
        pt_dense.bias = nn.Parameter(torch.Tensor(bias_weights.reshape(tuple(pt_dense.bias.shape))))

    activation = translate_activation(layer.activation)

    return pt_dense, activation


def translate_1d_locally_connected(layer: keras.layers.LocallyConnected1D) -> Tuple[nn.Module, Callable]:
    """Translate a 1-dimensional locally connected layer."""
    input_height, in_channels = layer.input.shape.as_list()[-2:]
    filters = layer.output.shape.as_list()[-1]
    kernel_size = layer.kernel_size[0]
    stride = layer.strides[0]

    keras_weights = layer.get_weights()
    if len(keras_weights) == 1:
        kernel_weights = keras_weights[0]
        bias_weights = None
        assert layer.use_bias is False
    elif len(keras_weights) == 2:
        kernel_weights, bias_weights = keras_weights
    else:
        raise NotImplementedError("Expected keras_weights to be of length 1 or 2. Was: {}. Full weights: {}".format(len(keras_weights), keras_weights))

    # TODO: Change to Conv1dLocal
    pt_local_conv = locally_connected.Conv2dLocal(
        in_height=input_height,
        in_width=1,
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        bias=layer.use_bias,
        data_format="channels_last"
    )

    shape = (pt_local_conv.out_height, pt_local_conv.out_width, filters, in_channels, kernel_size, 1)
    flat = torch.Tensor(kernel_weights.flatten())
    # The final shape used in the actual matrix multiplication.
    final = flat.view(pt_local_conv.out_height * pt_local_conv.out_width, in_channels * kernel_size * 1, filters)
    # Invert the transformations to obtain the required weights.
    reshaped = final.permute(0, 2, 1).view(shape)
    pt_local_conv.weight = nn.Parameter(reshaped)

    if bias_weights is not None:
        pt_local_conv.bias = Parameter(torch.Tensor(bias_weights).view(pt_local_conv.bias.shape))

    activation = translate_activation(layer.activation)

    return pt_local_conv, activation


def translate_2d_locally_connected(layer: keras.layers.LocallyConnected2D) -> Tuple[nn.Module, Callable]:
    """Translate a 2-dimensional locally connected layer."""
    # Extract various size and shape parameters
    if layer.data_format == "channels_first":
        in_channels, input_height, input_width = layer.input.shape.as_list()[-3:]
        filters, _, _ = layer.output.shape.as_list()[-3:]
    elif layer.data_format == "channels_last":
        input_height, input_width, in_channels = layer.input.shape.as_list()[-3:]
        _, _, filters = layer.output.shape.as_list()[-3:]
    else:
        raise NotImplementedError

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
        bias=layer.use_bias,
        data_format=layer.data_format
    )

    shape = (pt_local_conv.out_height, pt_local_conv.out_width, filters, in_channels, kernel_height, kernel_width)
    flat = torch.Tensor(kernel_weights.flatten())
    # The final shape used in the actual matrix multiplication.
    final = flat.view(pt_local_conv.out_height * pt_local_conv.out_width, in_channels * kernel_height * kernel_width, filters)
    # Invert the transformations to obtain the required weights.
    reshaped = final.permute(0, 2, 1).view(shape)

    pt_local_conv.weight = nn.Parameter(reshaped)

    if bias_weights is not None:
        reshaped_bias = bias_weights.reshape(pt_local_conv.bias.shape)
        pt_local_conv.bias = Parameter(torch.Tensor(reshaped_bias))

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
