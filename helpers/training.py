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
def train_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    parameters = {
        'bootstrap': [True],
        'min_samples_leaf': [2, 3, 4],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [50, 100, 150],
        'random_state': [42],
        'max_features': ['sqrt'],
        'n_jobs': [-1]
    }

    model = GridSearchCV(ensemble.RandomForestRegressor(), parameters, scoring='neg_mean_absolute_percentage_error', cv=5, n_jobs=-1)
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
        'min_samples_leaf': [2, 3, 4],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [50, 100, 150],
        'random_state': [42],
        'max_features': ['sqrt'],
        'n_jobs': [-1]
    }
    model = ensemble.GradientBoostingRegressor()
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
