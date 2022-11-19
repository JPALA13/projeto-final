import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import LabelPowerset
from stratified_cv import stratified_10fold_cv
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_top50.csv")
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
# Micro-Precision: 0.7397028073870007
# Micro-Recall: 0.970037389248656
# Micro-F1-measure: 0.8392642443186364 // 0.8393548698328863
# Macro-Precision: 0.7390678387977198
# Macro-Recall: 0.9695356468748614
# Macro-F1-measure: 0.8370494472644969 // 0.8387582270336835
# Hamming Loss: 0.2699235708367854
# 414.45051741600037

# Top 50
# Micro-Precision: 0.6182165080059033
# Micro-Recall: 0.5658833227081157
# Micro-F1-measure: 0.5902074439964885 // 0.5908934409566369
# Macro-Precision: 0.5919585027941634
# Macro-Recall: 0.5410707530537715
# Macro-F1-measure: 0.5628831616307697 // 0.5653718670198359
# Hamming Loss: 0.39549063549428726
# 4674.954325675964

# Top 100
# Micro-Precision: 0.5478152388111774
# Micro-Recall: 0.4826864173612317
# Micro-F1-measure: 0.5122968687249652 // 0.513192722037538
# Macro-Precision: 0.5058010281537744
# Macro-Recall: 0.44279043095788884
# Macro-F1-measure: 0.46851667581003265 // 0.47220297649504595
# Hamming Loss: 0.36829906394087775
# 5509.328054904938

# Order 10
# Micro-Precision: 0.6232293277716373
# Micro-Recall: 0.6709642589759527
# Micro-F1-measure: 0.6453242594702118 // 0.6462164677098396
# Macro-Precision: 0.6238719931968311
# Macro-Recall: 0.6713164521301184
# Macro-F1-measure: 0.6451834177691274 // 0.6467252461482834
# Hamming Loss: 0.3674646532887512
# 686.7165443897247
