"""Test converting full models"""
import pandas as pd

import torch

import ipdb
import mhcflurry

from typing import List
from pathlib import Path

from ktp import translate, models, mhcflurry_model

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

    # for act, blt, tor in zip(networks, ensemble.keras_models, ensemble.models):
    #     anet = act.network()
    #     bnet = blt.network()
    #     for w1, w2 in zip(anet.get_weights(), bnet.get_weights()):
    #         assert (w1 == w2).all()
    #     ipdb.set_trace()
    # mhcflurry_result = keras_model.predict([peptide], [allele])
    # print(mhcflurry_result)
    ipdb.set_trace()
