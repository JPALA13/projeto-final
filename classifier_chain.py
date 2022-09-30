from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
import time

start = time.time()

# Carga dos dados para memória
df = pd.read_csv("Data/SE_filter50.csv")
df = df.drop('CID', axis=1)
X = df.iloc[:,:9096]
Y = df.iloc[:,9096:]

# X, Y = make_multilabel_classification(n_samples=1384, n_features=9096, n_classes=644, random_state=1)

# n_samples, n_features = 1384, 9096
# n_classes = 644

stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.25, 0.75])
train_indexes, test_indexes = next(stratifier.split(X, Y))
X_treino, Y_treino = X.iloc[train_indexes, :], Y.iloc[train_indexes, :]
X_teste, Y_teste = X.iloc[test_indexes, :], Y.iloc[test_indexes, :]

# X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, random_state=1)

# print(X_treino.shape, X_teste.shape, Y_treino.shape, Y_teste.shape)

forest = RandomForestClassifier(random_state=1)
chain = ClassifierChain(forest, order='random', random_state=1)
chain.fit(X_treino, Y_treino)
predicao = chain.predict(X_teste)
resultado = chain.score(X_teste, Y_teste)
print(resultado)

end = time.time()
print(end - start)