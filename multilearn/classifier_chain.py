import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
from stratified_cv import stratified_10fold_cv
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

# Order Top 10

