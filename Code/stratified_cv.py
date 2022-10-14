from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from skmultilearn.model_selection import IterativeStratification
import numpy as np

def stratified_10fold_cv(estimator, X, Y):
    precision = [0] * 10
    recall = [0] * 10
    f_measure = [0] * 10
    loss_hamming = [0] * 10

    k_fold = IterativeStratification(n_splits=10, order=1)
    for index, (train_indexes, test_indexes) in enumerate(k_fold.split(X, Y)):
        print(index)
        
        X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
        X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

        estimator.fit(X_treino, Y_treino)
        Y_pred = estimator.predict(X_teste)
        
        precision[index] = precision_score(Y_teste, Y_pred, average='micro')
        recall[index] = recall_score(Y_teste, Y_pred, average='micro')
        f_measure[index] = f1_score(Y_teste, Y_pred, average='micro')
        loss_hamming[index] = hamming_loss(Y_teste, Y_pred)
    
    return {"Micro-Precision": np.mean(precision),
            "Micro-Recall": np.mean(recall),
            "Micro-F1-measure": np.mean(f_measure),
            "Hamming Loss": np.mean(loss_hamming)}
