import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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

multi_target_forest = MultiOutputClassifier(RandomForestClassifier(random_state=1), n_jobs=4)
results = stratified_10fold_cv(multi_target_forest, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Micro-Precision: 0.5468860383943943
# Micro-Recall: 0.31686167516044383
# Micro-F1-measure: 0.3999283311295253
# Hamming Loss: 0.13169221552246532
# 4126.507145404816
