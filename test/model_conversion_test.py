"""Test converting full models"""
import keras
import json
import pandas as pd
import numpy as np

import torch

import ipdb

from pathlib import Path

from ktp import translate, models, mhcflurry_model

test_dir = Path("./test")
data_dir = test_dir / Path("mhcflurry_data")
weights_dir = data_dir / Path("weights")
man = pd.read_csv("test/mhcflurry_data/manifest.csv")


def test_mhcflurry_models():
    """Tests full models from MHCFlurry"""
    row = man.iloc[0]
    name = row["model_name"]
    weights_path = weights_dir / Path("weights_{}.npz".format(name))
    weights_np = np.load(str(weights_path))
    weights = [weights_np["array_{}".format(i)] for i, _ in enumerate(weights_np)]
    weights_np.close()
    model_json = json.loads(row["config_json"])
    network_str = model_json['network_json']
    model = keras.models.model_from_json(network_str)
    model.set_weights(weights)
    pt_local1, local_act1 = translate.translate_1d_locally_connected(model.layers[1])
    pt_local2, local_act2 = translate.translate_1d_locally_connected(model.layers[2])
    pt_dense1, dense_act1 = translate.translate_fully_connected(model.layers[4])
    pt_dense2, dense_act2 = translate.translate_fully_connected(model.layers[5])


def test_predict():
    ensemble = mhcflurry_model.make_prediction("HLA-A*01:01", "SIINFEKL")
