import pandas as pd
from mlknn import MLkNN
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_order10.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

knn = MLkNN(k=9)
results = stratified_10fold_cv(knn, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Top 10
# Micro-Precision: 0.7529674049556224
# Micro-Recall: 0.8443393805874277
# Micro-F1-measure: 0.7948991192385966
# Hamming Loss: 0.3158120589533633
# 10.7027006149292

# Order 10
# Micro-Precision: 0.5773432504243499
# Micro-Recall: 0.6471616268082223
# Micro-F1-measure: 0.6095662397403467
# Hamming Loss: 0.4135936244062843
# 12.226706504821777

# Top 50


# Top 100

