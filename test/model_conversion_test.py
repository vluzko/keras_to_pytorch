"""Test converting full models"""
import keras
import json
import pandas as pd
import numpy as np

import torch

import ipdb

from pathlib import Path

from ktp import translate, models, mhcflurry_model
import mhcflurry
test_dir = Path("./test")
data_dir = test_dir / Path("mhcflurry_data")
weights_dir = data_dir / Path("weights")
man = pd.read_csv("test/mhcflurry_data/manifest.csv")


def test_predict():
    allele = "HLA-A*01:01"
    peptide = "SIINFEKL"
    prediction = mhcflurry_model.make_prediction(allele, peptide)
    print(prediction)
    ensemble = mhcflurry_model.get_predictor(allele)
    encoded_peptide = mhcflurry_model.peptides_to_network_input((peptide,), encoding="BLOSUM62")
    ensemble.predict
