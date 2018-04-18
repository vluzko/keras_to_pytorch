"""MHCFlurry in PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F
import json
from typing import NamedTuple, Tuple
import ipdb

class_key = 'class_name'


class MHCFlurryNet(nn.Module):
    """A PyTorch implementation of a MHCFlurry model
    All models take a 15 x 21 input tensor. (20 possible amino acids + 1 "null" amino acid, arranged in a 15mer sequence).

    Currently these models don't train correctly:
    * Weight initialization is different from MHCFlurry
    * No regularization is performed

    Attributes:
        layers (nn.ModuleList): The modules layers.
        transformations (List[Callable]): The modules transformations.
    """
    input_size = (15, 21)
    total_input_size = 15 * 21

    def __init__(self, allele: str, layers, transformations):
        super().__init__()
        self.layers = layers
        self.transformations = transformations
        # Input layer
        # Locally connected layer

    def forward(self, x):
        """Run the network."""
        for layer, transformation in zip(self.layers, self.transformations):
            x = layer(x)
            x = transformation(x)
        return x

    @staticmethod
    def layer_sizes(json_string: str):
        layers = MHCFlurryNet.layers(json_string)
        sizes = []
        for layer in layers:
            layer_class = layer['class_name']
            if layer_class == "InputLayer":
                size = input_layer_size(layer)
            elif layer_class == "LocallyConnected1D":
                size = locally_connected_layer_size(layer)
            elif layer_class == "Flatten":
                size = flatten_layer_size(layer)
            elif layer_class == "Dense":
                size = dense_layer_size(layer)
            sizes.append(size)
        return tuple(sizes)

    @staticmethod
    def load_from_json(json_string: str):
        """Load a model and its weights.

        Args:
            json_string:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def layer_names(json_string: str):
        layers = MHCFlurryNet.layers(json_string)
        return tuple(x['class_name'] for x in layers)

    @staticmethod
    def layers(json_string):
        network = MHCFlurryNet.network(json_string)
        layers = network['config']['layers']
        return layers

    @staticmethod
    def network(json_string):
        json_dict = MHCFlurryNet.from_mhcflurry_json(json_string)
        return json.loads(json_dict['network_json'])

    @staticmethod
    def from_mhcflurry_json(json_string: str):
        """Read a model from MHCflurry"""
        json_dict = json.loads(json_string)
        network = json.loads(json_dict['network_json'])
        layers = network['config']['layers']
        return json_dict


class NoLocal(MHCFlurryNet):
    """A MHCFlurry model with no locally connected layers"""

    architecture = ('InputLayer', 'Flatten', 'Dense', 'Dense')

    def __init__(self, allele: str, first_linear_out: int,):
        self.first_linear = nn.Linear(self.total_input_size, first_linear_out)
        self.second_linear = nn.Linear(first_linear_out, 1)

        self.first_transformation = F.tanh
        self.second_transformation = F.sigmoid

        layers = nn.ModuleList([self.first_linear, self.second_linear])
        transformations = (self.first_transformation, self.second_transformation)
        super().__init__(allele, layers, transformations)

    @staticmethod
    def load_from_json(json_string: str):
        raise NotImplementedError


class OneLocal(MHCFlurryNet):
    """A MHCFlurry model with one locally connected layer."""

    architecture = ('InputLayer', 'LocallyConnected1D', 'Flatten', 'Dense', 'Dense')

    def __init__(self, allele: str, first_linear_output: int):

        self.first_linear_layer = nn.Linear(0, first_linear_output)
        self.second_linear_layer = nn.Linear(first_linear_output, 1)

        self.first_linear_trans = F.tanh
        self.second_linear_trans = F.sigmoid

        super().__init__(allele, (), ())
        raise NotImplementedError

    @staticmethod
    def load_from_json(json_string: str):
        raise NotImplementedError


class TwoLocal(MHCFlurryNet):
    """A MHCFlurry model with two locally connected layers."""

    architecture = ('InputLayer', 'LocallyConnected1D', 'LocallyConnected1D', 'Flatten', 'Dense', 'Dense')

    def __init__(self, allele: str, first_linear_output: int):

        self.first_linear_layer = nn.Linear(0, first_linear_output)
        self.second_linear_layer = nn.Linear(first_linear_output, 1)

        self.first_linear_trans = F.tanh
        self.second_linear_trans = F.sigmoid

        super().__init__(allele, (), ())
        raise NotImplementedError

    @staticmethod
    def load_from_json(json_string: str):
        raise NotImplementedError


class MHCFlurryEnsemble(object):
    """An ensemble of (usually eight) MHCFlurry nets.

    Attributes:
    """

    def __init__(self, allele, models: Tuple[MHCFlurryNet]):
        pass

    @staticmethod
    def load_all(allele: str):
        """Create all the"""
        pass


def from_json(json_string: str):
    layer_names = MHCFlurryNet.layer_names(json_string)

    if layer_names == NoLocal.architecture:
        return NoLocal.load_from_json(json_string)
    elif layer_names == OneLocal.architecture:
        return OneLocal.load_from_json(json_string)
    elif layer_names == TwoLocal.architecture:
        return TwoLocal.load_from_json(json_string)
    else:
        assert False


class Kernel(NamedTuple):
    class_name: str
    scale: int
    reg: tuple


def kernel_keys(layer_config: dict):
    init = layer_config['kernel_initializer']
    reg = layer_config['kernel_regularizer']
    if reg is None:
        kern_reg = None
    else:
        kern_reg = (
            reg[class_key],
            reg['config']['l1'],
            reg['config']['l2']
        )
    assert init['config']['distribution'] == "uniform"
    assert layer_config['kernel_constraint'] is None
    return Kernel(
        init[class_key],
        init['config']['scale'],
        kern_reg
    )


def bias_keys(layer_config: dict):
    init = layer_config['bias_initializer']
    assert init['class_name'] == 'Zeros'
    assert layer_config['bias_regularizer'] is None
    assert layer_config['bias_constraint'] is None
    return (
        'Zeros',
    )


def input_layer_size(layer: dict):
    columns = ["input_size"]
    return tuple(layer['config']['batch_input_shape'])


class LocallyConnectedLayer(NamedTuple):
    filters: int
    kernel_size: tuple
    strides: tuple
    activation: str
    use_bias: bool
    kernel: tuple
    bias: tuple


def locally_connected_layer_size(layer: dict):
    config = layer['config']
    assert config['activity_regularizer'] is None
    return LocallyConnectedLayer(
        config['filters'],
        tuple(config['kernel_size']),
        tuple(config['strides']),
        config['activation'],
        config['use_bias'],
        kernel_keys(config),
        bias_keys(config)
    )


def flatten_layer_size(layer: dict):
    return ()


class DenseLayer(NamedTuple):
    units: int
    activation: str
    use_bias: bool
    kernel: tuple
    bias: tuple


def dense_layer_size(layer: dict):
    config = layer['config']
    return DenseLayer(
        config['units'],
        config['activation'],
        config['use_bias'],
        kernel_keys(config),
        bias_keys(config)
    )
