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
    alleles = man[man['allele'].str.startswith('HLA')]['allele'].unique()
    peptide = "SIINFEKL"
    encoded_peptide = mhcflurry_model.peptides_to_network_input((peptide,), encoding="BLOSUM62")
    loaded_keras_model: mhcflurry.Class1AffinityPredictor = mhcflurry.Class1AffinityPredictor.load()
    all_torch = mhcflurry_model.AllEnsembles().to(device)
    total = len(alleles)
    for i, allele in enumerate(alleles):
        print("Run {} of {}".format(i, total))
        #ensemble = mhcflurry_model.get_predictor(allele)
        #networks: List[mhcflurry.Class1NeuralNetwork] = loaded_keras_model.allele_to_allele_specific_models[allele]
        # mhc = ensemble.keras_models[0].predict([peptide])
        tinput = torch.Tensor(encoded_peptide.reshape((1, 15, 1, 21))).to(device)
        #torch_pred = ensemble.forward(torch.Tensor(encoded_peptide.reshape((1, 15, 1, 21))))
        torch_pred = all_torch.forward(tinput, alleles=(allele,)).cpu().data.numpy()
        keras_pred = loaded_keras_model.predict(allele=allele, peptides=[peptide])
        print(torch_pred)
        print(keras_pred)
        assert np.isclose(torch_pred, keras_pred)

        #for keras_model, torch_model, loaded_net in zip(ensemble.keras_models, ensemble.models, networks):
       #      anet = loaded_net.network()
       #      bnet = keras_model.network()
       #      aout = anet.predict(encoded_peptide)
       #      kout = bnet.predict(encoded_peptide)
       #      tin = torch.Tensor(encoded_peptide).reshape(1, 15, 1, 21)
       #      tout = torch_model(tin).cpu().data.numpy().reshape(kout.shape)
       #      assert np.isclose(kout, tout) and np.isclose(aout, tout)
       # break

    # ipdb.set_trace()
        # ipdb.set_trace()
    #     i = 0
    #     for keras_layer in bnet.layers:
    #         if not isinstance(keras_layer, keras.layers.LocallyConnected1D) and not isinstance(keras_layer, keras.layers.Dense):
    #             continue
    #
    #         torch_layer = torch_model.layers[i].to(device)
    #         torch_act = torch_model.activations[i]
    #         input_shape = keras_layer.input_shape[1:]
    #         inputs = keras.Input(input_shape)
    #         keras_model = keras.Model(inputs, keras_layer(inputs))
    #         keras_input = np.random.uniform(-100, 100, (1, *input_shape)).astype(np.float32)
    #         keras_output = keras_model.predict(keras_input)
    #
    #         if isinstance(torch_layer, locally_connected.Conv2dLocal):
    #             keras_input = keras_input.reshape((1, *torch_layer.input_shape.numpy().astype(int)))
    #
    #         torch_input = torch.Tensor(keras_input).to(device)
    #         after_mult = torch_layer(torch_input)
    #         torch_output = torch_act(after_mult).cpu().data.numpy().reshape(keras_output.shape)
    #         assert np.isclose(keras_output, torch_output, atol=1e-4, rtol=1e-4).all()
    #         i += 1

    # mhcflurry_result = keras_model.predict([peptide], [allele])
    # print(mhcflurry_result)


def compare_layers(keras_layer, torch_layer, input_shape):
    """Compare a Keras and a PyTorch layer."""
    keras_input = np.random.uniform(-100, 100, input_shape).astype(np.float32)
    keras_output = keras_layer.call(keras_input)

    if isinstance(torch_layer, locally_connected.Conv2dLocal):
        keras_input = keras_input.reshape((1, *torch_layer.input_shape))

    torch_input = torch.Tensor(keras_input).to(device)
    torch_output = torch_layer(torch_input).cpu().data.numpy()
    assert np.isclose(keras_output, torch_output).all()
