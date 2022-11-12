from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from skmultilearn.model_selection import IterativeStratification
import numpy as np

def stratified_10fold_cv(estimator, X, Y):
    size = 10

    micro_precision = np.zeros(size)
    micro_recall = np.zeros(size)
    micro_f_measure = np.zeros(size)
    macro_precision = np.zeros(size)
    macro_recall = np.zeros(size)
    macro_f_measure = np.zeros(size)
    loss_hamming = np.zeros(size)

    k_fold = IterativeStratification(n_splits=size, random_state=1)

    for index, (train_indexes, test_indexes) in enumerate(k_fold.split(X, Y)):
        print(index)
        
        X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
        X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

        estimator.fit(X_treino, Y_treino)
        Y_pred = estimator.predict(X_teste)
        
        micro_precision[index] = precision_score(Y_teste, Y_pred, average='micro')
        micro_recall[index] = recall_score(Y_teste, Y_pred, average='micro')
        micro_f_measure[index] = f1_score(Y_teste, Y_pred, average='micro')
        macro_precision[index] = precision_score(Y_teste, Y_pred, average='macro')
        macro_recall[index] = recall_score(Y_teste, Y_pred, average='macro')
        macro_f_measure[index] = f1_score(Y_teste, Y_pred, average='macro')
        loss_hamming[index] = hamming_loss(Y_teste, Y_pred)
    
    return {"Micro-Precision": np.mean(micro_precision),
            "Micro-Recall": np.mean(micro_recall),
            "Micro-F1-measure": np.mean(micro_f_measure),
            "Macro-Precision": np.mean(macro_precision),
            "Macro-Recall": np.mean(macro_recall),
            "Macro-F1-measure": np.mean(macro_f_measure),
            "Hamming Loss": np.mean(loss_hamming)}
