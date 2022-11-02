import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from stratified_cv import stratified_10fold_cv
import numpy as np
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_top10.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

forest = RandomForestClassifier(random_state=1, n_estimators=500)
binary_relevance = BinaryRelevance(forest, True)

results = stratified_10fold_cv(binary_relevance, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Top 10
# Micro-Precision: 0.7634759195335852
# Micro-Recall: 0.9183882209033316
# Micro-F1-measure: 0.8333727640467439
# Hamming Loss: 0.26650533839238155
# 1410.4593708515167

# Order Top 10

