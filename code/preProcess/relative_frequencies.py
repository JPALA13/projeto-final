import pandas as pd

df = pd.read_csv("../data/se_frequencies.csv", header=None, names=['ID', 'Name', 'Frequency'])
df['Frequency'] = df['Frequency'] / 1394

df['Frequency'] = df['Frequency'].apply(lambda x: f"{x:.2%}")

# df.to_csv("../Data/se_relative_frequencies.csv", index=False, header=False)
print(df)
