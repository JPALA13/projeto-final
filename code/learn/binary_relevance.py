import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from code.validation.stratified_cv import stratified_10fold_cv
import time

# setting path
sys.path.append('../code')

start = time.time()

# Carga dos dados para memoria
df = pd.read_csv("../data/SE_filter50_top.csv")
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
# Micro-Precision: 0.5533844490479403
# Micro-Recall: 0.32430040963222495
# Micro-F1-measure: 0.4076684847283073
# Macro-Precision: 0.28920050022214594
# Macro-Recall: 0.14481243202785893
# Macro-F1-measure: 0.17384523378441527
# Hamming Loss: 0.13073857435559366
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
# Micro-Precision: 0.5523301175659994
# Micro-Recall: 0.32126778873524736
# Micro-F1-measure: 0.40562050521192017
# Macro-Precision: 0.2861223904827458
# Macro-Recall: 0.14219563241380312
# Macro-F1-measure: 0.17098782823513636
# Hamming Loss: 0.13067339780602186
# 21645.646597385406
