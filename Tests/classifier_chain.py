import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from skmultilearn.model_selection import IterativeStratification
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("Data/SE_filter50.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# n_samples, n_features = 1394, 9096
# n_classes = 644

precision = [0] * 10
recall = [0] * 10
f_measure = [0] * 10
loss_hamming = [0] * 10

k_fold = IterativeStratification(n_splits=10, order=1)
chain = ClassifierChain(RandomForestClassifier(random_state=1), order='random', random_state=1)
for index, (train_indexes, test_indexes) in enumerate(k_fold.split(X, Y)):
    print(index)

    X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
    X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

    chain.fit(X_treino, Y_treino)
    Y_pred = chain.predict(X_teste)

    precision[index] = precision_score(Y_teste, Y_pred, average='micro')
    recall[index] = recall_score(Y_teste, Y_pred, average='micro')
    f_measure[index] = f1_score(Y_teste, Y_pred, average='micro')
    loss_hamming[index] = hamming_loss(Y_teste, Y_pred)

print("Micro-Precision:", np.mean(precision))
print("Micro-Recall:", np.mean(recall))
print("Micro-F1-measure:", np.mean(f_measure))
print("Hamming Loss:", np.mean(loss_hamming))

end = time.time()
print(end - start)

# Micro-Precision: 0.5406176669359307
# Micro-Recall: 0.16859672256430758
# Micro-F1-measure: 0.25565175472570034
# Hamming Loss: 0.13579216516249562
# 9818.942804574966
