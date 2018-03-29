"""Test converting full models"""
import pandas as pd
from collections import defaultdict
import ipdb

from pathlib import Path

from ktp import layers, models, mhcflurry_model

test_dir = Path("./test")


def test_mhcflurry_models():
    """Tests full models from MHCFlurry"""
    model_path = Path("full_mhcflurry_keras.json")
    json_string = (test_dir / model_path).open().read()
    mhcflurry_model.MHCFlurryNet.from_mhcflurry_json(json_string)


# test_mhcflurry_models()


def group_by_layers():
    man = pd.read_csv("test/mhcflurry_data/manifest.csv")
    print("Number of models: {}".format(len(man)))
    layer_groups = defaultdict(list)
    for i, row in man.iterrows():
        layer_names = mhcflurry_model.MHCFlurryNet.layer_names(row.config_json)
        layer_groups[layer_names].append(i)
    return layer_groups, man


def check_layer_sizes():
    layer_groups, man = group_by_layers()
    for layer_names, indices in layer_groups.items():
        relevant_rows = man.loc[indices]
        sizes = set()
        for i, row in relevant_rows.iterrows():
            size = mhcflurry_model.MHCFlurryNet.layer_sizes(row.config_json)
            sizes.add(size)
            break
        print("\n\nOverall architecture: {}".format(layer_names))
        print("Differences:")
        for size in sizes:
            print("One group: {}".format(size))


def check_layers():
    man = pd.read_csv("test/mhcflurry_data/manifest.csv")
    layer_sets = set()
    for i, row in man.iterrows():
        layer_names = mhcflurry_model.MHCFlurryNet.layer_names(row.config_json)
        layer_sets.add(layer_names)
    print(layer_sets)


def main():
    check_layer_sizes()


if __name__ == "__main__":
    main()
