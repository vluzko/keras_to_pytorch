import keras
import torch
import json
f_name = "test/HLA-A*01:01.json"


def input_layer(layer):
    pass


def flatten_layer(layer):
    pass


def dense_layer(layer):
    pass


def keras_layer_to_pytorch_layer(layer: dict):

    print(layer)


def load_from_json(model: dict):
    network = model['network_json']['config']
    layers = network['layers']
    for layer in layers:
        keras_layer_to_pytorch_layer(layer)


def main():
    with open(f_name, "r") as f:
        model = json.loads(f.read())
        load_from_json(model)


if __name__ == "__main__":
    main()
