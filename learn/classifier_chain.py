import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_top644.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

forest = RandomForestClassifier(random_state=1, n_estimators=500)
chain = ClassifierChain(forest, random_state=1)
results = stratified_10fold_cv(chain, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Micro-Precision: 0.5527377673639794
# Micro-Recall: 0.21180096823555022
# Micro-F1-measure: 0.30502843249742356
# Hamming Loss: 0.13370104618957185
# 38350.799241542816

# Micro-Precision: 0.3820605760982242
# Micro-Recall: 0.47329762531460784
# Micro-F1-measure: 0.4225262029283014
# Hamming Loss: 0.17953282464702663
