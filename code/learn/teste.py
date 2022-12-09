import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import multilabel_confusion_matrix

# Carga dos dados para memoria
df = pd.read_csv("../../data/SE_filter50.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:, :9096]
Y = df.iloc[:, 9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

k_fold = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.25, 0.75])
train_indexes, test_indexes = next(k_fold.split(X, Y))

X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

forest = RandomForestClassifier(random_state=1, n_estimators=500)

binary_relevance = MultiOutputClassifier(forest, n_jobs=4)

estimators = [binary_relevance]

results = []
for estimator in estimators:
    print(estimator)

    estimator.fit(X_treino, Y_treino)
    Y_pred = estimator.predict(X_teste)
    results.append(Y_pred)

    print(f'Micro-Precision: {precision_score(Y_teste, Y_pred, average="micro")}') # TP/(TP+FP)
    print(f'Micro-Recall: {recall_score(Y_teste, Y_pred, average="micro")}') # TP/(TP+FN)
    print(f'Micro-F1-measure: {f1_score(Y_teste, Y_pred, average="micro")}') # 2*P*R/(P+R)
    print(f'Hamming Loss: {hamming_loss(Y_teste, Y_pred)}') # 1 - ACC

    print()
    matriz = multilabel_confusion_matrix(Y_teste, Y_pred)
    print(f'TN: {matriz[:, 0, 0].sum()}')
    print(f'FP: {matriz[:, 0, 1].sum()}')
    print(f'FN: {matriz[:, 1, 0].sum()}')
    print(f'TP: {matriz[:, 1, 1].sum()}')
    print()
    print(matriz)
    print()

# print(results[0]==results[1])

# a = np.array([[[3, 1],
#                [0, 2]],
              
#               [[5, 0],
#                [1, 0]],

#               [[2, 1],
#                [1, 2]]])
# print(a[:, 1, 1].sum())

# Micro-Precision: 0.49275302812564153
# Micro-Recall: 0.36221779548472777
# Micro-F1-measure: 0.4175204828917843
# Hamming Loss: 0.15203316261668665 

# TN: 174762
# FP: 12354
# FN: 21131
# TP: 12001
