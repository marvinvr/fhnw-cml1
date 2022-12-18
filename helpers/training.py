# Training functions
from sklearn import linear_model, ensemble, neural_network
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import pandas as pd
import numpy as np


## Linear Models
### Linear Regression
def train_linear_regression(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    parameters = {}

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         linear_model.LinearRegression)


## Ensemble
### Random Forest
def train_random_forest(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> dict:
    parameters = {
        'min_samples_split': [9, 10, 11, 12],
        'min_samples_leaf': [6, 7, 8, 9],
        'max_features': ['sqrt'],
        'n_estimators': [80, 100, 120],
        'random_state': [42]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         ensemble.RandomForestRegressor)


### Gradient Boosting
def train_gradient_boosting(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    parameters = {
        'tree_method': ['gpu_hist'],
        'gpu_id': [0],

        'max_depth': [20],
        'max_leaves': [0],
        'n_estimators': [110],
        'seed': [42],
        'lambda': [1.3],
        'alpha': [0.05]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         XGBRegressor, 1)


def train_gradient_boosting_robust(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    parameters = {
        'loss': ['absolute_error'],
        'max_depth': [17],
        'min_samples_split': [2],
        'min_samples_leaf': [20, 21, 22],
        'max_features': ['sqrt'],
        'n_estimators': [120],
        'random_state': [42]
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         ensemble.GradientBoostingRegressor, -1)

def train_gradient_boosting_v1(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series) -> dict:
    '''parameters = {
        'loss': ['absolute_error'],
        'max_depth': [17, 19, 21, 23, 25],  #[26],
        'min_samples_split': [1, 2],
        'min_samples_leaf': [12, 13, 15, 17],
        'max_features': ['sqrt'],
        'n_estimators': [120, 130, 140],
        'random_state': [42]
    }'''
    parameters = {
        'loss': ['absolute_error'],
        'max_depth': [21],  # [26],
        'min_samples_split': [1],
        'min_samples_leaf': [13],
        'max_features': ['sqrt'],
        'n_estimators': [130],
        'random_state': [42]
    }


    # NOT WORKING, TRANSFORMED KAGGLE

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         ensemble.GradientBoostingRegressor, -1)


## Neural Network
### MLP Regressor
def train_mlp_regressor(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> dict:
    parameters = {
        'hidden_layer_sizes': [
            #tuple((int(np.sqrt(len(X_train.columns))) for _ in range(3))),
            tuple((int(np.sqrt(len(X_train.columns))) for _ in range(4)))
        ],
        #'activation': ['logistic', 'relu'],
        #'alpha': [0.0001, 0.0005, 0.001],
        'learning_rate': ['adaptive'],
        #'learning_rate_init': [0.001, 0.005],
        'max_iter': [10_000],
    }

    return _run_training(X_train, X_test, y_train, y_test,
                         parameters,
                         neural_network.MLPRegressor,
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

    model = GridSearchCV(ModelClass(),
                         parameters,
                         scoring='neg_mean_absolute_percentage_error',
                         cv=cv,
                         n_jobs=n_jobs,
                         verbose=10)
    model.fit(X_train, y_train)

    return {
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }
