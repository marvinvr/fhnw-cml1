# Training functions
from functools import reduce

from sklearn import neighbors, ensemble, neural_network
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np


## Neighbors
### K Neighbors
def train_k_neighbors(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    parameters = {
        'n_neighbors': [5],
        'weights': ['distance'],
        'leaf_size': [10],
        'p': [2]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         neighbors.KNeighborsClassifier)


## Ensemble
### Random Forest
def train_random_forest(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> dict:
    parameters = {
        'min_samples_split': [10],
        'max_features': [0.5],
        'min_samples_leaf': [10],
        'n_estimators': [130],
        'random_state': [42]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         ensemble.RandomForestClassifier)


### Gradient Boosting
def train_gradient_boosting(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    parameters = {
        'max_depth': [2],
        'min_samples_split': [10],
        'max_features': [0.5],
        'min_samples_leaf': [10],
        'n_estimators': [130],
        'random_state': [42]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         ensemble.GradientBoostingClassifier, -1)


## Neural Network
### MLP Classifier
def train_mlp_classifier(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> dict:
    parameters = {
        'hidden_layer_sizes': [
            tuple((int(np.sqrt(len(X_train.columns))) for _ in range(4)))
        ],
        'activation': ['relu'],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.005],
        'max_iter': [10_000],
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         neural_network.MLPClassifier,
                         1,
                         1)


def _run_training(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,
                  parameters: dict,
                  ModelClass: callable,
                  n_jobs: int = -1,
                  cv: int = 5
                  ) -> dict:
    print(f'Training {ModelClass.__name__} with {n_jobs} jobs')
    print(f'Parameters: {parameters}')

    if not all(len(l) < 2 for l in parameters.values()):
        grid_search = GridSearchCV(ModelClass(),
                             parameters,
                             scoring='f1_weighted',
                             cv=cv,
                             n_jobs=n_jobs,
                             verbose=10)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    else:
        best_params = reduce(
            lambda state, key: {**state, key: parameters[key][0]},
            parameters.keys(),
            {})

    model = ModelClass(**best_params)
    model.fit(X_train, y_train)

    return {
        "num_columns": len(X_train.columns),
        "score": f1_score(y_test, model.predict(X_test), average='weighted'),
        "best_params": best_params,
        "model": model
    }
