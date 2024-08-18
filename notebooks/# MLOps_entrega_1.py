# MLOps_entrega_1

# %%
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma Decision Tree
clf = DecisionTreeClassifier(random_state=42)

# Treinar a Decision Tree
clf.fit(X_train, y_train)

# Prever as classes para o conjunto de teste
y_pred = clf.predict(X_test)

# Imprimir a precisão da Decision Tree
print("Precisão:", clf.score(X_test, y_test))

# %% [markdown]
# 

# %% [markdown]
# Exercício 1-  Treinamento do modelo baseado em árvore de decisão

# %%

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset MNIST
digits = load_digits(n_class=10)
X = digits.data
y = digits.target

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Criar uma Decision Tree
clf = DecisionTreeClassifier(random_state=30)

# Treinar a Decision Tree
clf.fit(X_train, y_train)

# Prever as classes para o conjunto de teste
y_pred = clf.predict(X_test)

# Imprimir a precisão da Decision Tree
print("Precisão Sklearn:", clf.score(X_test, y_test))

# %%
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset Digits
digits = load_digits(n_class=10)
X = digits.data
y = digits.target

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Criar uma Decision Tree com parâmetros personalizados
clf = DecisionTreeClassifier(
    max_depth=12,              # Tente aumentar ou diminuir
    min_samples_split=40,      # Aumente para evitar overfitting
    min_samples_leaf=10,       # Aumente para garantir generalização
    max_features=None,         # Pode tentar valores como 'sqrt' ou um número fixo
    random_state=30
)

# Treinar a Decision Tree
clf.fit(X_train, y_train)

# Prever as classes para o conjunto de teste
y_pred = clf.predict(X_test)

# Imprimir a precisão da Decision Tree
print("Precisão ajustada:", clf.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV

# Definindo o espaço de parâmetros para testar
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 20, 40],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Criando o GridSearchCV para encontrar os melhores parâmetros
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=30), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
print("Melhores parâmetros:", grid_search.best_params_)

# Treinando e avaliando o modelo com os melhores parâmetros
best_clf = grid_search.best_estimator_
print("Precisão ajustada com melhores parâmetros:", best_clf.score(X_test, y_test))

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset
digits = load_digits()
X = digits.data
y = digits.target

# Dividir o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Criar o modelo RandomForest
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=30)

# Treinar o modelo
random_forest_clf.fit(X_train, y_train)

# Avaliar o modelo
accuracy_rf = random_forest_clf.score(X_test, y_test)
print("Precisão do RandomForest:", accuracy_rf)

# %%
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o dataset
digits = load_digits()
X = digits.data
y = digits.target

# Dividir o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Criar o modelo XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss')

# Treinar o modelo
xgb_clf.fit(X_train, y_train)

# Prever as classes para o conjunto de teste
y_pred = xgb_clf.predict(X_test)

# Avaliar o modelo
accuracy_xgb = accuracy_score(y_test, y_pred)
print("Precisão do XGBoost:", accuracy_xgb)

# %%
!git add .

# %%
!git commit -m "Atualizacoes"

# %%
!git push

# %% [markdown]
# Exercício 2 - Avaliação dos ganhos com a utilização de modelos Ensemble

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset Digits
digits = load_digits()
X = digits.data
y = digits.target

# Dividir o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Criar e treinar a árvore de decisão
dt_clf = DecisionTreeClassifier(random_state=30)
dt_clf.fit(X_train, y_train)
dt_accuracy = dt_clf.score(X_test, y_test)

# Criar e treinar o modelo RandomForest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=30)
rf_clf.fit(X_train, y_train)
rf_accuracy = rf_clf.score(X_test, y_test)

# Criar e treinar o modelo XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=30)
xgb_clf.fit(X_train, y_train)
xgb_accuracy = xgb_clf.predict(X_test)

# Avaliar o modelo
xgb_accuracy = xgb_clf.score(X_test, y_test)

print("Precisão da Árvore de Decisão:", dt_accuracy)
print("Precisão do Random Forest:", rf_accuracy)
print("Precisão do XGBoost:", xgb_accuracy)



# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Árvore de Decisão
dt_pred = dt_clf.predict(X_test)
dt_proba = dt_clf.predict_proba(X_test)[:, 1]  # Probabilidades para a classe positiva

# Random Forest
rf_pred = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)[:, 1]

# XGBoost
xgb_pred = xgb_clf.predict(X_test)
xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("Relatório de Classificação da Árvore de Decisão:")
print(classification_report(y_test, dt_pred))

print("Relatório de Classificação do Random Forest:")
print(classification_report(y_test, rf_pred))

print("Relatório de Classificação do XGBoost:")
print(classification_report(y_test, xgb_pred))


# %% [markdown]
# Exercício 3 :  Visualização da árvore de decisão e Medida de Impureza

# %%
plot_tree(clf, filled=True)
plt.show()

# %%
import numpy as np

def gini_impurity(labels):
    # Verifica se o array está vazio usando .size
    if labels.size == 0:
        return 0
    label_counts = np.unique(labels, return_counts=True)[1]
    probabilities = label_counts / label_counts.sum()
    return 1 - np.sum(np.square(probabilities))

def entropy(labels):
    # Verifica se o array está vazio usando .size
    if labels.size == 0:
        return 0
    label_counts = np.unique(labels, return_counts=True)[1]
    probabilities = label_counts / label_counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Exemplo de cálculo para um subconjunto dos dados
subset = y_train[:50]  # Suponha uma divisão de dados hipotética
print("Impureza de Gini:", gini_impurity(subset))
print("Entropia:", entropy(subset))

# %%



