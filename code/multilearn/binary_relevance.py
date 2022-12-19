import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from ucd.code.ud_forest import DFERandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from validation.stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memoria
df = pd.read_csv("../../data/SE_top10.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:, :9096]
Y = df.iloc[:, 9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

# forest = RandomForestClassifier(random_state=1, n_estimators=500)
forest = DFERandomForestClassifier(random_state=1, n_estimators=500)
binary_relevance = BinaryRelevance(forest, True)

results = stratified_10fold_cv(binary_relevance, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Top 10
# Micro-Precision: 0.7621704674242097
# Micro-Recall: 0.9160307096044182
# Micro-F1-measure: 0.8319382660853417 // 0.8320475085713994
# Macro-Precision: 0.7592150411941085
# Macro-Recall: 0.9132162132015653
# Macro-F1-measure: 0.8283723271067099 // 0.8291252427897077
# Hamming Loss: 0.26875443839507634
# 1414.1738951206207

# Top 50
# Micro-Precision: 0.651812598044587
# Micro-Recall: 0.7152274203524883
# Micro-F1-measure: 0.6816040112901366 // 0.6820491525907487
# Macro-Precision: 0.62841948653816
# Macro-Recall: 0.6825685328545628
# Macro-F1-measure: 0.6513162342221444 // 0.6543757236504164
# Hamming Loss: 0.33689582300950577
# 7035.585059404373

# Top 100
# Micro-Precision: 0.6152290302039188
# Micro-Recall: 0.5883662619734229
# Micro-F1-measure: 0.6010796424514135 // 0.6014978740965006
# Macro-Precision: 0.5728601600285171
# Macro-Recall: 0.5242768648832914
# Macro-F1-measure: 0.5362241298227277 // 0.5474928325209582
# Hamming Loss: 0.31341330193875017
# 16069.545104265213

# Order 10
# Micro-Precision: 0.6200558449094502
# Micro-Recall: 0.7002388011460012
# Micro-F1-measure: 0.6571007567959484 // 0.6577125231555756
# Macro-Precision: 0.6207980753259431
# Macro-Recall: 0.7001253998051056
# Macro-F1-measure: 0.656743884766073 // 0.6580797583943243
# Hamming Loss: 0.3645453717166475
# 1262.8424417972565
