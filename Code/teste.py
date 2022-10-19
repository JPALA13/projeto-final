import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import multilabel_confusion_matrix

# Carga dos dados para memória
df = pd.read_csv("../Data/SE_filter50.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

k_fold = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.25, 0.75])
train_indexes, test_indexes = next(k_fold.split(X, Y))

X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

forest = RandomForestClassifier(random_state=1, n_estimators=500)

multi_target_forest = MultiOutputClassifier(forest, n_jobs=4)
binary_relevance = OneVsRestClassifier(forest, n_jobs=4)

estimators = [multi_target_forest, binary_relevance]

results = []
for estimator in estimators:
    print(estimator)

    estimator.fit(X_treino, Y_treino)
    Y_pred = estimator.predict(X_teste)
    results.append(Y_pred)

    print(f'Micro-Precision: {precision_score(Y_teste, Y_pred, average="micro")}')
    print(f'Micro-Recall: {recall_score(Y_teste, Y_pred, average="micro")}')
    print(f'Micro-F1-measure: {f1_score(Y_teste, Y_pred, average="micro")}')
    print(f'Hamming Loss: {hamming_loss(Y_teste, Y_pred)}')

    print()
    print(multilabel_confusion_matrix(Y_teste, Y_pred))

    print()

print(results[0]==results[1])
