"""Test converting full models"""
import pandas as pd
import numpy as np
import pytest
import torch

import mhcflurry

from pathlib import Path

from ktp import translate, models, mhcflurry_model, locally_connected

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = Path("./test")
data_dir = test_dir / Path("mhcflurry_data")
weights_dir = data_dir / Path("weights")
man = pd.read_csv("test/mhcflurry_data/manifest.csv")


@pytest.mark.skip()
def test_predict():
    alleles = man[man['allele'].str.startswith('HLA')]['allele'].unique()
    # peptide = "SIINFEKL"

    sample_peptides = (
        'RDAVILLM', 'VYEAADMI', 'RLLSPTTIV', 'LAYTIGTTHF', 'FPVTPQVPV', 'KSLFNTVATL', 'GLVILLVLAL', 'SLFGGMSWI', 'RDWAHNSL', 'TSAVLLLLVV', 'MLVQSCTSI', 'LLDAHIPQL', 'SLYNTVAAL',
        'AMKADIQHV', 'RPPIFIRRLH', 'GHQAAMQML', 'GLIMVLSFL', 'SNVSAALAAL', 'LFNWAVKTKL', 'SMNATLVQA', 'SLFNAVAVL', 'LWEWASVRF', 'LFLNTLSFV', 'TPGPGIRYPL', 'FISGIQYLA',
        'NMIIMDEAHF', 'CQTYKWETF', 'QMWKCLIRL', 'CPRRPAVAF', 'SLNFLGGTTV', 'SPKRLATAI', 'FAFSDLCIVY', 'AVCTRGVAK', 'KAKKTPMGF', 'GLMNNAFEWI', 'AIRGSVTPAV', 'VTEHDTLLY',
        'IPITAAAWYL', 'QRLHGLSAF', 'IENSSVNVSL', 'RDWAHNGL', 'FYGMWPLLL', 'VLSDFKSWL', 'NEGCGWMGW', 'PTIAGAGDV', 'FVRSSNLKF', 'LWLTDNTHI', 'LLDEQGVGPL', 'SLFNTVAVL', 'VLFGLLCLL'
    )
    encoded_peptides = tuple(mhcflurry_model.peptides_to_network_input((x, ), encoding="BLOSUM62") for x in sample_peptides)
    # encoded_peptide = mhcflurry_model.peptides_to_network_input((peptide,), encoding="BLOSUM62")
    loaded_keras_model: mhcflurry.Class1AffinityPredictor = mhcflurry.Class1AffinityPredictor.load()
    all_torch = mhcflurry_model.AllEnsembles().to(device)

    results = {a: [] for a in alleles}

    total = len(alleles)
    for i, allele in enumerate(alleles):
        print("Run {} of {}".format(i, total))
        df = pd.read_csv("data/results_{}.csv".format(allele))
        for peptide, encoding in zip(sample_peptides, encoded_peptides):
            try:
                existing = df[df["peptide"] == peptide]
            except:
                existing = []
            if len(existing) > 0:
                assert len(existing) == 1
                torch_pred = existing.iloc[0]['torchMHC_prediction']
                keras_pred = existing.iloc[0]['MHCFlurry_prediction']
            else:
                tinput = torch.Tensor(encoding.reshape((1, 15, 1, 21))).to(device)
                torch_pred = all_torch.forward(tinput, alleles=(allele,)).cpu().data.numpy()[0][0]
                keras_pred = loaded_keras_model.predict(allele=allele, peptides=[peptide])[0]
                assert np.isclose(torch_pred, keras_pred)
            results[allele].append((peptide, torch_pred, keras_pred))

    for allele in alleles:
        df = pd.DataFrame.from_records(results[allele])
        df.columns = ["peptide", "torchMHC_prediction", "MHCFlurry_prediction"]
        df.to_csv("data/results_{}.csv".format(allele))

import ipdb
def test_batch_predict():
    alleles = man[man['allele'].str.startswith('HLA')]['allele'].unique()
    # peptide = "SIINFEKL"

    sample_peptides = (
        'RDAVILLM', 'VYEAADMI', 'RLLSPTTIV', 'LAYTIGTTHF', 'FPVTPQVPV', 'KSLFNTVATL', 'GLVILLVLAL', 'SLFGGMSWI', 'RDWAHNSL', 'TSAVLLLLVV', 'MLVQSCTSI', 'LLDAHIPQL', 'SLYNTVAAL',
    )
    encoded_peptides = tuple(mhcflurry_model.peptides_to_network_input((x, ), encoding="BLOSUM62") for x in sample_peptides)
    as_tensors = tuple(torch.Tensor(x).view(15, 1, 21) for x in encoded_peptides)
    batch = torch.stack(as_tensors).to(device)
    # encoded_peptide = mhcflurry_model.peptides_to_network_input((peptide,), encoding="BLOSUM62")
    loaded_keras_model: mhcflurry.Class1AffinityPredictor = mhcflurry.Class1AffinityPredictor.load()
    all_torch = mhcflurry_model.AllEnsembles().to(device)

    # results = {a: [] for a in alleles}
    allele = alleles[0]

    result = all_torch(batch, alleles=(allele,))

    # total = len(alleles)
    # for i, allele in enumerate(alleles):
        # print("Run {} of {}".format(i, total))
        # for peptide, encoding in zip(sample_peptides, encoded_peptides):
        #     tinput = torch.Tensor(encoding.reshape((1, 15, 1, 21))).to(device)
        #     torch_pred = all_torch.forward(tinput, alleles=(allele,)).cpu().data.numpy()[0][0]
        #     keras_pred = loaded_keras_model.predict(allele=allele, peptides=[peptide])[0]
        #     assert np.isclose(torch_pred, keras_pred)
        #     results[allele].append((peptide, torch_pred, keras_pred))

        # break
