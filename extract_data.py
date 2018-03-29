import pandas as pd

man = pd.read_csv("test/mhcflurry_data/manifest.csv")

for i, row in man.iterrows():
    name = row.allele
    if name[:3] != "HLA":
        continue
    # keras_config = json.loads(row.config_json)['network_json']
    with open("test/{}_keras.json".format(name), "w") as f:
        f.write(row.config_json)
    break
