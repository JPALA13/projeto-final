import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
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
classifier_chain = ClassifierChain(forest, True)
results = stratified_10fold_cv(classifier_chain, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Top 10
# Micro-Precision: 0.7490020254643077
# Micro-Recall: 0.9438241767265776
# Micro-F1-measure: 0.8350347801668949
# Hamming Loss: 0.270838943725462
# 831.9310872554779

# Order 10
# Micro-Precision: 0.6222922389934015
# Micro-Recall: 0.6847875052145518
# Micro-F1-measure: 0.6517239668768557
# Hamming Loss: 0.3650264147207872
# 799.8484697341919

# Top 50
# Micro-Precision: 0.5392082573461388
# Micro-Recall: 0.9096471743948635
# Micro-F1-measure: 0.6761750877990695
# Hamming Loss: 0.43910204182347956
# 3348.16602230072

# Top 100
# Micro-Precision: 0.44457908612068964
# Micro-Recall: 0.8843833396885534
# Micro-F1-measure: 0.5909770211135531
# Hamming Loss: 0.4909609765216619
# 5953.670835733414
