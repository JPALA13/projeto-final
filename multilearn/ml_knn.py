import pandas as pd
from skmultilearn.adapt import MLkNN
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memoria
df = pd.read_csv("../Data/SE_top100.csv")
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
# Micro-Precision: 0.7549810814670035
# Micro-Recall: 0.8381173696887396
# Micro-F1-measure: 0.7931558217071556 // 0.7943799803519179
# Macro-Precision: 0.7523902627174366
# Macro-Recall: 0.8334053492786065
# Macro-F1-measure: 0.7857315421542606 // 0.7908283576400921
# Hamming Loss: 0.31555024263226417
# 11.068220853805542

# Top 50
# Micro-Precision: 0.6149001751366211
# Micro-Recall: 0.6454059705109968
# Micro-F1-measure: 0.6291863562632565 // 0.6297838754051354
# Macro-Precision: 0.5966319378641999
# Macro-Recall: 0.6161333847661306
# Macro-F1-measure: 0.5950866095034805 // 0.6062258682307273
# Hamming Loss: 0.38297735050632864
# 20.206759691238403

# Top 100
# Micro-Precision: 0.5754405927782901
# Micro-Recall: 0.5185472240670935
# Micro-F1-measure: 0.5450377085258997 // 0.5455145247616188
# Macro-Precision: 0.5365144895699916
# Macro-Recall: 0.45638631260289664
# Macro-F1-measure: 0.4702785604195855 // 0.4932171854771812
# Hamming Loss: 0.3471103981121464
# 29.314936876296997

# Order 10
# Micro-Precision: 0.5792738003891383
# Micro-Recall: 0.614778023285543
# Micro-F1-measure: 0.5943452510271526 // 0.5964980663039708
# Macro-Precision: 0.5858815283683393
# Macro-Recall: 0.6140518768046015
# Macro-F1-measure: 0.5876995221098061 // 0.5996360306810133
# Hamming Loss: 0.41712367908450976
# 10.427119016647339
