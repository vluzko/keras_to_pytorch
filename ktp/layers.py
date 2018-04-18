import keras
import torch
from functools import singledispatch


@singledispatch
def translate_layer(layer_json):
    """Translate a single layer."""
    pass
