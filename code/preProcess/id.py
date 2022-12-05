import pandas as pd

df = pd.read_csv("../data/SE_filter50.csv")
df = df.iloc[:, :1]

df.to_csv("../Data/side_effects_id.csv", index=False)
print(df)
