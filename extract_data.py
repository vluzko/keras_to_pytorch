import pandas as pd
import json

man = pd.read_csv("test/mhcflurry_data/manifest.csv")

for i, row in man.iterrows():
    name = row.allele
    if name[:3] != "HLA":
        continue
    model = json.loads(row.config_json)
    model['network_json'] = json.loads(model['network_json'])
    as_string = json.dumps(model)
    with open("test/{}.json".format(name), "w") as f:
        f.write(as_string)
