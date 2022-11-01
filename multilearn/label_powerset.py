import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from Code.stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_filter50_order.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

forest = RandomForestClassifier(random_state=1, n_estimators=500)
label_powerset = LabelPowerset(forest, True)
results = stratified_10fold_cv(label_powerset, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)


