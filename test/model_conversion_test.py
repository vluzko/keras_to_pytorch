"""Test converting full models"""
import pandas as pd
import numpy as np
import keras
import torch

import ipdb
import mhcflurry

from typing import List
from pathlib import Path

from ktp import translate, models, mhcflurry_model, locally_connected

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = Path("./test")
data_dir = test_dir / Path("mhcflurry_data")
weights_dir = data_dir / Path("weights")
man = pd.read_csv("test/mhcflurry_data/manifest.csv")


def test_predict():
    allele = "HLA-A*01:01"
    peptide = "SIINFEKL"
    # prediction = mhcflurry_model.make_prediction(allele, peptide)
    # print(prediction)
    ensemble = mhcflurry_model.get_predictor(allele)
    encoded_peptide = mhcflurry_model.peptides_to_network_input((peptide,), encoding="BLOSUM62")
    # keras_model: mhcflurry.Class1AffinityPredictor = mhcflurry.Class1AffinityPredictor.load()
    # networks: List[mhcflurry.Class1NeuralNetwork] = keras_model.allele_to_allele_specific_models[allele]

    mhc = ensemble.keras_models[0].predict([peptide])
    pred = ensemble.forward(torch.Tensor(encoded_peptide.reshape((1, 15, 1, 21))))

    for keras_model, torch_model in zip(ensemble.keras_models, ensemble.models):
        bnet = keras_model.network()
        i = 0
        for keras_layer in bnet.layers:
            if not isinstance(keras_layer, keras.layers.LocallyConnected1D) and not isinstance(keras_layer, keras.layers.Dense):
                continue

            torch_layer = torch_model.layers[i]
            input_shape = keras_layer.input_shape[1:]
            inputs = keras.Input(input_shape)
            keras_model = keras.Model(inputs, keras_layer(inputs))
            keras_input = np.random.uniform(-100, 100, (1, *input_shape)).astype(np.float32)
            keras_output = keras_model.predict(keras_input)

            if isinstance(torch_layer, locally_connected.Conv2dLocal):
                keras_input = keras_input.reshape((1, *torch_layer.input_shape.numpy().astype(int)))

            torch_input = torch.Tensor(keras_input).to(device)
            torch_output = torch_layer(torch_input).cpu().data.numpy()
            assert np.isclose(keras_output, torch_output).all()
            i += 1

    # mhcflurry_result = keras_model.predict([peptide], [allele])
    # print(mhcflurry_result)
    ipdb.set_trace()


def compare_layers(keras_layer, torch_layer, input_shape):
    """Compare a Keras and a PyTorch layer."""
    keras_input = np.random.uniform(-100, 100, input_shape).astype(np.float32)
    keras_output = keras_layer.call(keras_input)

    if isinstance(torch_layer, locally_connected.Conv2dLocal):
        keras_input = keras_input.reshape((1, *torch_layer.input_shape))

    torch_input = torch.Tensor(keras_input).to(device)
    torch_output = torch_layer(torch_input).cpu().data.numpy()
    assert np.isclose(keras_output, torch_output).all()
