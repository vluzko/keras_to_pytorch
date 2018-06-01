"""MHCFlurry in PyTorch."""
import json
import keras
import collections
import math
import numpy as np
import pandas as pd

from copy import copy
from io import StringIO
from typing import NamedTuple, Tuple
from pathlib import Path
from functools import reduce, partial
from scipy.stats.mstats import gmean

import torch
from torch import nn

from ktp import translate
import ipdb

import mhcflurry

class_key = 'class_name'
test_dir = Path("./test")
data_dir = test_dir / Path("mhcflurry_data")
weights_dir = data_dir / Path("weights")
manifest_path = data_dir / Path("manifest.csv")


class MHCFlurryNet(nn.Module):
    """A PyTorch implementation of a MHCFlurry model
    All models take a 15 x 21 input tensor. (20 possible amino acids + 1 "null" amino acid, arranged in a 15mer sequence).

    Currently these models don't train correctly:
    * Weight initialization is different from MHCFlurry
    * No regularization is performed

    Attributes:
        layers (nn.ModuleList): The modules layers.
        activations (List[Callable]): The modules activations.
    """
    input_size = (15, 21)
    total_input_size = 15 * 21

    def __init__(self, allele: str, layers, activations):
        super().__init__()
        self.allele = allele
        self.layers = nn.ModuleList(layers)
        self.activations = activations

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        raise NotImplementedError

    @classmethod
    def from_model(cls, allele: str, model):
        """Create a net from a keras model."""
        raise NotImplementedError


class NoLocal(MHCFlurryNet):
    """A MHCFlurry model with no locally connected layers"""

    architecture = ('InputLayer', 'Flatten', 'Dense', 'Dense')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.view(-1)
        for layer, act in zip(self.layers, self.activations):
            output = act(layer(output))
        return output

    @classmethod
    def from_model(cls, allele: str, model: keras.models.Model) -> 'NoLocal':
        dense1, act1 = translate.translate_fully_connected(model.layers[2])
        dense2, act2 = translate.translate_fully_connected(model.layers[3])
        return NoLocal(allele, [dense1, dense2], [act1, act2])


class OneLocal(MHCFlurryNet):
    """A MHCFlurry model with one locally connected layer."""

    architecture = ('InputLayer', 'LocallyConnected1D', 'Flatten', 'Dense', 'Dense')

    def forward(self, input: torch.Tensor):
        output = input
        for layer, act in zip(self.layers[:1], self.activations[:1]):
            output = act(layer(output))
        output = output.view(-1)
        for layer, act in zip(self.layers[1:], self.activations[1:]):
            output = act(layer(output))
        return output

    @classmethod
    def from_model(cls, allele: str, model: keras.models.Model) -> 'OneLocal':
        local1, act1 = translate.translate_1d_locally_connected(model.layers[1])
        dense1, act2 = translate.translate_fully_connected(model.layers[3])
        dense2, act3 = translate.translate_fully_connected(model.layers[4])
        return OneLocal(allele, [local1, dense1, dense2], [act1, act2, act3])


class TwoLocal(MHCFlurryNet):
    """A MHCFlurry model with two locally connected layers."""

    architecture = ('InputLayer', 'LocallyConnected1D', 'LocallyConnected1D', 'Flatten', 'Dense', 'Dense')

    def forward(self, input: torch.Tensor):
        """Two locally connected layers, then two dense layers."""
        output = input
        for layer, act in zip(self.layers[:2], self.activations[:2]):
            output = act(layer(output))
        output = output.view(-1)
        for layer, act in zip(self.layers[2:], self.activations[2:]):
            output = act(layer(output))
        return output

    @classmethod
    def from_model(cls, allele: str, model: keras.models.Model) -> 'TwoLocal':
        local1, act1 = translate.translate_1d_locally_connected(model.layers[1])
        local2, act2 = translate.translate_1d_locally_connected(model.layers[2])
        dense1, act3 = translate.translate_fully_connected(model.layers[4])
        dense2, act4 = translate.translate_fully_connected(model.layers[5])
        return TwoLocal(allele, [local1, local2, dense1, dense2], [act1, act2, act3, act4])


class MHCFlurryEnsemble(nn.Module):
    """An ensemble of (usually eight) MHCFlurry nets.

    Attributes:
        keras_models (Tuple[mhcflurry.Class1NeuralNetwork, ...]): The keras networks this ensemble was produced from.
    """

    def __init__(self, allele, models: Tuple[MHCFlurryNet, ...], keras_models: Tuple[mhcflurry.Class1NeuralNetwork, ...]):
        super().__init__()
        self.keras_models = keras_models
        self.allele = allele
        self.models = nn.ModuleList(models)
        self.exp = 1 / len(self.models)

    def forward(self, input):
        """Compute the geometric mean of the outputs of the individual models."""
        outputs = torch.Tensor(tuple(to_ic50(x(input)) for x in self.models))
        log_sum = outputs.log().sum()
        root = log_sum * self.exp
        geometric_mean = root.exp()
        return geometric_mean

    def keras_pred(self, input):
        """Make a prediction with the underlying keras models."""
        preds = [k.predict(input) for k in self.keras_models]
        geometric_mean = reduce(lambda a, b: a*b, preds, 1) ** self.exp
        return preds, geometric_mean

    @classmethod
    def ensemble_from_rows(cls, allele, model_rows: pd.DataFrame):
        """Create an ensemble model from the dataframe.
        Does not check that all rows define models for the same allele.
        """
        all_models = tuple(row_to_net(allele, row) for _, row in model_rows.iterrows())
        models, keras_models = zip(*all_models)
        return cls(allele, models, keras_models)


class AllEnsembles(nn.Module):

    def __init__(self):
        super().__init__()
        manifest = pd.read_csv(str(manifest_path))
        print(1)
        self.alleles = {x for x in manifest['allele'] if x.startswith('HLA')}
        self.ensemble_indices = {x: i for i, x in enumerate(self.alleles)}
        self.ensembles = nn.ModuleList(get_predictor(x) for x in self.alleles)
        
    def forward(input, alleles: Tuple[str, ...]=()):
        if alleles == ():
            alleles = self.alleles

        results = torch.Tensor(len(alleles)).view(len(alleles), 1)
        for i, allele in enumerate(alleles):
            ensemble_index = self.ensemble_indices[allele]
            ensemble = self.ensembles[ensemble_index]
            pred = ensemble(input)
            results[i] = pred
        return results


def row_to_net(allele, row: pd.Series) -> Tuple[MHCFlurryNet, mhcflurry.Class1NeuralNetwork]:
    """Convert a row of the manifest to a PyTorch model and a keras model"""
    name = row["model_name"]
    weights_path = weights_dir / Path("weights_{}.npz".format(name))
    model_json = json.loads(row["config_json"])
    network_str = model_json["network_json"]

    mhc = mhcflurry.Class1NeuralNetwork.from_config(model_json, weights_loader=partial(mhcflurry.Class1AffinityPredictor.load_weights, str(weights_path.resolve())))

    cls = model_architecture(network_str)
    pytorch_model = cls.from_model(allele, mhc.network())
    return pytorch_model, mhc


def model_architecture(network_json_string: str) -> type(MHCFlurryNet):
    """Get the sequence of layers in the network."""
    network_json = json.loads(network_json_string)
    layer_json = network_json["config"]["layers"]
    layer_names = tuple(x["class_name"] for x in layer_json)
    return architecture_maps[layer_names]


def to_ic50(x, max_ic50=50000.0):
    """Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].

    Args:
        x: numpy.array of float
        max_ic50:

    Returns:
        numpy.array of float
    """
    return max_ic50 ** (1.0 - x)


def get_predictor(allele: str) -> MHCFlurryEnsemble:
    manifest = pd.read_csv(str(manifest_path))
    relevant_rows = manifest[manifest['allele'] == allele]
    ensemble = MHCFlurryEnsemble.ensemble_from_rows(allele, relevant_rows)
    return ensemble


def make_prediction(allele: str, peptide: str) -> float:
    """Predict the binding affinity of the given allele and peptide.
    Uses an ensemble of all stored models for that allele.
    """
    ensemble = get_predictor(allele)
    encoded_peptide = peptides_to_network_input((peptide,), encoding="BLOSUM62")
    return ensemble(torch.Tensor(encoded_peptide.reshape((1, 15, 1, 21))))


def peptides_to_network_input(peptides: Tuple[str, ...], encoding: str, kmer_size: int = 15) -> np.ndarray:
    """Encode peptides to the fixed-length encoding expected by the neural network (which depends on the architecture).

    Args:
        peptides: tuple of peptide strings
        encoding: The way to encode the peptides.
        kmer_size: The size of the network input.


    Returns:
        numpy array of encoded peptides.
    """
    seq_array = np.array(peptides)
    fixed_length_sequences = sequences_to_fixed_length_index_encoded_array(seq_array, left_edge=4, right_edge=4, max_length=15)
    if encoding == "embedding":
        encoded = fixed_length_sequences
    elif encoding in ENCODING_DATA_FRAMES.keys():
        encoded = fixed_vectors_encoding(fixed_length_sequences, ENCODING_DATA_FRAMES[encoding])
        assert encoded.shape[0] == len(seq_array)
    else:
        raise ValueError("Unsupported peptide_amino_acid_encoding: {}".format(encoding))
    assert len(encoded) == len(peptides)
    return encoded


COMMON_AMINO_ACIDS = collections.OrderedDict(sorted({
                                                        "A": "Alanine",
                                                        "R": "Arginine",
                                                        "N": "Asparagine",
                                                        "D": "Aspartic Acid",
                                                        "C": "Cysteine",
                                                        "E": "Glutamic Acid",
                                                        "Q": "Glutamine",
                                                        "G": "Glycine",
                                                        "H": "Histidine",
                                                        "I": "Isoleucine",
                                                        "L": "Leucine",
                                                        "K": "Lysine",
                                                        "M": "Methionine",
                                                        "F": "Phenylalanine",
                                                        "P": "Proline",
                                                        "S": "Serine",
                                                        "T": "Threonine",
                                                        "W": "Tryptophan",
                                                        "Y": "Tyrosine",
                                                        "V": "Valine",
                                                    }.items()))
COMMON_AMINO_ACIDS_WITH_UNKNOWN = copy(COMMON_AMINO_ACIDS)
COMMON_AMINO_ACIDS_WITH_UNKNOWN["X"] = "Unknown"

AMINO_ACID_INDEX = dict(
    (letter, i) for (i, letter) in enumerate(COMMON_AMINO_ACIDS_WITH_UNKNOWN))

AMINO_ACIDS = list(COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())

# Why isn't this a numerical matrix? The world may never know.
BLOSUM62_MATRIX = pd.read_table(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS]
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all()

ENCODING_DATA_FRAMES = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS)
}


def sequences_to_fixed_length_index_encoded_array(sequences, left_edge=4, right_edge=4, max_length=15) -> np.ndarray:
    """
    Transform a sequence of strings, where each string is of length at least
    left_edge + right_edge and at most max_length into strings of length
    max_length using a scheme designed to preserve the anchor positions of
    class I peptides.

    The first left_edge characters in the input always map to the first
    left_edge characters in the output. Similarly for the last right_edge
    characters. The middle characters are filled in based on the length,
    with the X character filling in the blanks.

    For example, using defaults:

    AAAACDDDD -> AAAAXXXCXXXDDDD

    The strings are also converted to int categorical amino acid indices.

    Args:
        sequences : string
        left_edge : int
        right_edge : int
        max_length : int

    Returns:
        numpy array of shape (len(sequences), max_length) and dtype int
    """

    # Result array is int32, filled with X (null amino acid) value.
    result = np.full(
        fill_value=AMINO_ACID_INDEX['X'],
        shape=(len(sequences), max_length),
        dtype="int32")

    df = pd.DataFrame({"peptide": sequences})
    df["length"] = df.peptide.str.len()

    middle_length = max_length - left_edge - right_edge

    # For efficiency we handle each supported peptide length using bulk
    # array operations.
    for (length, sub_df) in df.groupby("length"):
        if length < left_edge + right_edge:
            raise ValueError(
                "Sequence '%s' (length %d) unsupported: length must be at "
                "least %d. There are %d total peptides with this length." % (
                    sub_df.iloc[0].peptide, length, left_edge + right_edge,
                    len(sub_df))
            )
        elif length > max_length:
            raise ValueError(
                "Sequence '%s' (length %d) unsupported: length must be at "
                "most %d. There are %d total peptides with this length." % (
                    sub_df.iloc[0].peptide, length, max_length,
                    len(sub_df))
            )

        # Array of shape (num peptides, length) giving fixed-length amino
        # acid encoding each peptide of the current length.
        fixed_length_sequences = np.stack(
            sub_df.peptide.map(
                lambda s: np.array([
                    AMINO_ACID_INDEX[char] for char in s
                ])).values)

        num_null = max_length - length
        num_null_left = int(math.ceil(num_null / 2))
        num_middle_filled = middle_length - num_null
        middle_start = left_edge + num_null_left

        # Set left edge
        result[sub_df.index, :left_edge] = fixed_length_sequences[:, :left_edge]

        # Set middle.
        result[sub_df.index, middle_start: middle_start + num_middle_filled] = fixed_length_sequences[:, left_edge: left_edge + num_middle_filled]

        # Set right edge.
        result[sub_df.index, -right_edge:] = fixed_length_sequences[:, -right_edge:]
    return result


def fixed_vectors_encoding(index_encoded_sequences, letter_to_vector_df):
    """
    Given a `n` x `k` matrix of integers such as that returned by `index_encoding()` and
    a dataframe mapping each index to an arbitrary vector, return a `n * k * m`
    array where the (`i`, `j`)'th element is `letter_to_vector_df.iloc[sequence[i][j]]`.

    The dataframe index and columns names are ignored here; the indexing is done
    entirely by integer position in the dataframe.

    Parameters
    ----------
    index_encoded_sequences : `n` x `k` array of integers

    letter_to_vector_df : pandas.DataFrame of shape (`alphabet size`, `m`)

    Returns
    -------
    numpy.array of integers with shape (`n`, `k`, `m`)
    """
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    target_shape = (
        num_sequences, sequence_length, letter_to_vector_df.shape[0])
    result = letter_to_vector_df.iloc[
        index_encoded_sequences.flatten()
    ].values.reshape(target_shape)
    return result


architecture_maps = {x.architecture: x for x in (NoLocal, OneLocal, TwoLocal)}
