import pandas as pd
import numpy as np

df = pd.read_csv("../data/SE.csv")

new = df.iloc[:, :9097]

se = np.sum(df.iloc[:, 9097:])
order_se = se.sort_values(ascending=False).index

new[order_se] = df[order_se]
new = new.iloc[:, :9197] # 9107, 9147, 9197

print(new.iloc[:, 9097:])
# new.to_csv("../Data/SE_top100.csv", index=False)
