"""Convert individual layers."""
import keras
import torch
from functools import singledispatch


@singledispatch
def translate_layer(layer_json):
    """Translate a single layer."""
    pass


def translate_fully_connected(layer: keras.layers.Dense):
    """Translate a fully connected layer."""
    pass


def translate_1d_locally_connected(layer: keras.layers.LocallyConnected1D):
    """Translate a 1-dimensional locally connected layer."""
    _, input_height, in_channels = layer.input.shape.as_list()
    _, _, filters = layer.output.shape.as_list()
    kernel_weights, bias_weights = layer.get_weights()
