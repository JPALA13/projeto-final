import pandas as pd
import numpy as np

df = pd.read_csv("../Data/SE.csv")
half = df.shape[0]/2

new = df.iloc[:, :9097]

se = np.sum(df.iloc[:, 9097:])
order_se = se.sort_values(ascending=False).index

new[order_se] = df[order_se]
df = new.iloc[:, :9197]

#

new = df.iloc[:, :9097]

se = abs(np.sum(df.iloc[:, 9097:]) - half)
order_se = se.sort_values().index

new[order_se] = df[order_se]

print(new.iloc[:, 9097:])
# new.to_csv("../Data/SE_top100_order.csv", index=False)
