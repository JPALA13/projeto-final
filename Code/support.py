import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
results = stratified_10fold_cv(forest, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# tree.DecisionTreeClassifier
# tree.ExtraTreeClassifier
# ensemble.ExtraTreesClassifier
# neighbors.KNeighborsClassifier
# neural_network.MLPClassifier
# neighbors.RadiusNeighborsClassifier
# ensemble.RandomForestClassifier
# linear_model.RidgeClassifier
# linear_model.RidgeClassifierCV

# Micro-Precision: 0.6641639564341002
# Micro-Recall: 0.22248232024117948
# Micro-F1-measure: 0.33304088146523175
# Hamming Loss: 0.123434380164483
# 5048.662222623825
