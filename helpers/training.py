# Training functions
from sklearn import linear_model, ensemble, neural_network
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


## Linear Models
### Linear Regression
def train_linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }

## Ensemble

### Random Forest
def train_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    parameters = {
        'criterion': ['gini', 'log_loss'],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [10, 15, 20],
        'max_features': ['sqrt'],
        'n_estimators': [50, 100, 150],
    }

    model = GridSearchCV(ensemble.RandomForestRegressor(), parameters, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)

    return {
        "columns": list(X_test.columns),
        "num_columns": len(X_test.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }

### Gradient Boosting
def train_gradient_boosting(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    parameters = {
        'loss': ['absolute_error'],
        'criterion': ['friedman_mse'],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [10, 15, 20],
        'max_features': ['sqrt'],
        'n_estimators': [50, 100, 150],
    }
    model = GridSearchCV(ensemble.GradientBoostingRegressor(), parameters, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }


## Neural Network

### MLP Regressor
def train_mlp_regressor(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    parameters = {
        'hidden_layer_sizes': [(np.sqrt(len(X_train.columns)) for _ in range(3)), np.sqrt(len(X_train.columns)) for _ in range(4)],
        'activation': ['logistic', 'relu'],
        'alpha': [0.0001, 0.0005, 0.001],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.001, 0.005],
        'max_iter': [10_000, 20_000],
    }


    model = GridSearchCV(neural_network.MLPRegressor(), parameters, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }
