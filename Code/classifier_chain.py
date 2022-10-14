import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_filter50.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

chain = ClassifierChain(RandomForestClassifier(random_state=1), order='random', random_state=1)
results = stratified_10fold_cv(chain, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Micro-Precision: 0.550350149316611
# Micro-Recall: 0.17092100419134956
# Micro-F1-measure: 0.25942130276352493
# Hamming Loss: 0.13546468632763617
# 6071.369469642639
