# Training functions
from sklearn import linear_model, ensemble, neural_network
from sklearn.metrics import mean_absolute_percentage_error
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
def train_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model = ensemble.RandomForestRegressor(max_features='sqrt', n_jobs=-1, random_state=2**12)
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
    model = neural_network.MLPRegressor(max_iter=15000)
    model.fit(X_train, y_train)

    return {
        "columns": list(X_train.columns),
        "num_columns": len(X_train.columns),
        "score": mean_absolute_percentage_error(y_test, model.predict(X_test)),
        "model": model
    }
