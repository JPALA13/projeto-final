import pandas as pd

df = pd.read_csv("../data/SE_top10.csv")
df = df.iloc[:, :1]
stitch_id = df['CID'].values

names = pd.read_csv("../data/drug_names.tsv", sep='\t', header=None, names=['CID', 'name'])
for i, name in enumerate(names.values):
    if name[0] not in stitch_id:
        names.drop(i, inplace=True)

names.to_csv("../Data/side_effects_id_name.csv", index=False)
print(names)
