# Training functions
from sklearn import linear_model, ensemble, neural_network
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import pandas as pd


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


### Ridge
def train_ridge(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model = linear_model.Ridge()
    model.fit(X_train, y_train, 100)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }


### Bayesian Regression
def train_bayesian_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                              y_test: pd.Series) -> dict:
    model = linear_model.BayesianRidge()
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }


### Passive Agressive Regressor
def train_passive_agressive(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model = linear_model.PassiveAggressiveRegressor()
    model.fit(X_train, y_train, )

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }


### Quantile Regression
def train_quantile_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                              y_test: pd.Series) -> dict:
    model = linear_model.BayesianRidge()
    model.fit(X_train, y_train, 100)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }


## Ensemble

### Random Forest
def train_random_forest(X: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict:
    parameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [1, 2],
        'n_estimators': [100, 200, 300, 1000]
    }

    model = GridSearchCV(ensemble.RandomForestRegressor(), parameters, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1)
    model.fit(X, y)

    best_model = model.best_estimator_
    best_model.fit(X_train, y_train)

    return {
        "columns": list(X.columns),
        "num_columns": len(y.columns),
        "score": mean_absolute_percentage_error(y_test, best_model.predict(X_test)),
        "model": model
    }


## Neural Network

### MLP Regressor
def train_mlp_regressor(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model = neural_network.MLPRegressor(max_iter=15000)
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }
