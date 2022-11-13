import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from stratified_cv import stratified_10fold_cv
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
binary_relevance = MultiOutputClassifier(forest, n_jobs=4)
results = stratified_10fold_cv(binary_relevance, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Random
# Micro-Precision: 0.5517972818544343
# Micro-Recall: 0.3210789127339513
# Micro-F1-measure: 0.40477014726659954
# Hamming Loss: 0.13085041708679013
# 21245.271294355392

# Order
# Micro-Precision: 0.5524449688736832
# Micro-Recall: 0.32306444162229775
# Micro-F1-measure: 0.4072532480890255
# Macro-Precision: 0.27741706597508703
# Macro-Recall: 0.14286943794929602
# Macro-F1-measure: 0.1708841002075862
# Hamming Loss: 0.13068598581878746
# 21104.044816732407

# Top

