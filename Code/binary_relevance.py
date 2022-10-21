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

forest = RandomForestClassifier(random_state=1, n_estimators=500)
binary_relevance = MultiOutputClassifier(forest, n_jobs=4)
results = stratified_10fold_cv(binary_relevance, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Micro-Precision: 0.5517972818544343
# Micro-Recall: 0.3210789127339513
# Micro-F1-measure: 0.40477014726659954
# Hamming Loss: 0.13085041708679013
# 21245.271294355392
