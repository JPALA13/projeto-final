import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_filter50_top.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

forest = RandomForestClassifier(random_state=1, n_estimators=500)
chain = ClassifierChain(forest, random_state=1)
results = stratified_10fold_cv(chain, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Random
# Micro-Precision: 0.5477454893748828
# Micro-Recall: 0.21309944657302066
# Micro-F1-measure: 0.3051982550754059
# Macro-Precision: 0.2759367312789252
# Macro-Recall: 0.10028503308252892
# Macro-F1-measure: 0.13672632928554002
# Hamming Loss: 0.1347333562043339
# 30661.67826652527

# Order
# Micro-Precision: 0.3882570029355401
# Micro-Recall: 0.46880693719575073
# Micro-F1-measure: 0.4239911639587029
# Macro-Precision: 0.26829223334228697
# Macro-Recall: 0.29265993426959247
# Macro-F1-measure: 0.24933522423021856
# Hamming Loss: 0.17666239442494097
# 26986.964843034744

# Top
# Micro-Precision: 0.31723039254589375
# Micro-Recall: 0.6087955726881614
# Micro-F1-measure: 0.416806956634771
# Macro-Precision: 0.239524345410566
# Macro-Recall: 0.35943691697098357
# Macro-F1-measure: 0.23980390529404735
# Hamming Loss: 0.2362381295726867
# 27866.70461177826
