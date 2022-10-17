import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
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

forest = RandomForestClassifier(random_state=1, n_estimators=500)
binary_relevance = OneVsRestClassifier(forest, n_jobs=4)
results = stratified_10fold_cv(binary_relevance, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Micro-Precision: 0.5489633277938697
# Micro-Recall: 0.3231076260755573
# Micro-F1-measure: 0.4060346139549906
# Hamming Loss: 0.1308195699515891
# 20800.935278892517
