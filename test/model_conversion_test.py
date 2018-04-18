"""Test converting full models"""
import keras
import ipdb

from pathlib import Path

from ktp import layers, models

test_dir = Path("./test")


def test_mhcflurry_models():
    """Tests full models from MHCFlurry"""
    model_path = Path("mhcflurry_keras_1.json")
    json_string = (test_dir / model_path).open().read()
    model = keras.models.model_from_json(json_string)
    ipdb.set_trace()
