import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from code.validation.stratified_cv import stratified_10fold_cv
import time

# setting path
sys.path.append('../code')

start = time.time()

# Carga dos dados para memoria
df = pd.read_csv("../data/SE_filter50_top.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:, :9096]
Y = df.iloc[:, 9096:]

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

# Random
# Micro-Precision: 0.662403493256279
# Micro-Recall: 0.2247277015801085
# Micro-F1-measure: 0.3351044252565954
# Macro-Precision: 0.2270044103351815
# Macro-Recall: 0.07712926461951924
# Macro-F1-measure: 0.09832185387889686
# Hamming Loss: 0.1236550432846408
# 4952.3666036129

# Order
# Micro-Precision: 0.6658936697134025
# Micro-Recall: 0.22363070033817842
# Micro-F1-measure: 0.3343877629203619
# Macro-Precision: 0.22190526169723895
# Macro-Recall: 0.0764513329961752
# Macro-F1-measure: 0.09721947681682161
# Hamming Loss: 0.12356943436473775
# 4860.986392736435

# Top
# Micro-Precision: 0.6647788613552268
# Micro-Recall: 0.22328552032959234
# Micro-F1-measure: 0.33410069925028507
# Macro-Precision: 0.22469066641321772
# Macro-Recall: 0.07668489795795437
# Macro-F1-measure: 0.09754014944905874
# Hamming Loss: 0.12350136202150297
# 4996.859210252762
