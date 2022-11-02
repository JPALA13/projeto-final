import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_top100.csv")
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

# Top 10
# Micro-Precision: 0.7404954937144472
# Micro-Recall: 0.9664366549087798
# Micro-F1-measure: 0.8383568883690057
# Hamming Loss: 0.27067477184098365
# 408.7702934741974

# Order 10
# Micro-Precision: 0.6225281335814561
# Micro-Recall: 0.6710178965311934
# Micro-F1-measure: 0.6446885325590596
# Hamming Loss: 0.3684461152882206
# 818.9287805557251

# Top 50
# Micro-Precision: 0.6198262062454042
# Micro-Recall: 0.5735569416301808
# Micro-F1-measure: 0.5949633175499927
# Hamming Loss: 0.39385579961824707
# 4637.660880804062

# Top 100

