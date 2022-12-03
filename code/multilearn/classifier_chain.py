import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
from code.validation.stratified_cv import stratified_10fold_cv
import time

# setting path
sys.path.append('../code')

start = time.time()

# Carga dos dados para memoria
df = pd.read_csv("../data/SE_top100.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:, :9096]
Y = df.iloc[:, 9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

forest = RandomForestClassifier(random_state=1, n_estimators=500)
classifier_chain = ClassifierChain(forest, True)
results = stratified_10fold_cv(classifier_chain, X, Y)

for k, v in results.items():
    print(f'{k}: {v}')

end = time.time()
print(end - start)

# Top 10
# Micro-Precision: 0.748348447517445
# Micro-Recall: 0.9461567006368521
# Micro-F1-measure: 0.8355927637674896 // 0.835706989501979
# Macro-Precision: 0.746978687307992
# Macro-Recall: 0.9450513213273204
# Macro-F1-measure: 0.833046893694285 // 0.8344216022659403
# Hamming Loss: 0.270412149366789
# 840.6297750473022

# Top 50
# Micro-Precision: 0.5390064272162228
# Micro-Recall: 0.9080627937694986
# Micro-F1-measure: 0.6760421294557867 // 0.6764730740721197
# Macro-Precision: 0.533527384244574
# Macro-Recall: 0.9018404100511843
# Macro-F1-measure: 0.6635463110992145 // 0.6704296374668691
# Hamming Loss: 0.4385806497834814
# 3368.351553440094

# Top 100
# Micro-Precision: 0.4421686098874054
# Micro-Recall: 0.8863914770122312
# Micro-F1-measure: 0.5895454351218963 // 0.5900139422691392
# Macro-Precision: 0.4343038255062718
# Macro-Recall: 0.8745805455477417
# Macro-F1-measure: 0.5699618376988589 // 0.5803930202617904
# Hamming Loss: 0.49506823289239466
# 6039.097524881363

# Order 10
# Micro-Precision: 0.622572385866316
# Micro-Recall: 0.6971702318989441
# Micro-F1-measure: 0.6570809739823398 // 0.6577630043701447
# Macro-Precision: 0.6232188022362182
# Macro-Recall: 0.6974162819549891
# Macro-F1-measure: 0.6569534480156708 // 0.6582332168862711
# Hamming Loss: 0.36284137808739436
# 826.7751734256744
