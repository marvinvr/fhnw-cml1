from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


def select_features(X, y, threshold = 0.01) -> list:
    model = Lasso()
    model.fit(X, y)

    sfm = SelectFromModel(model, threshold=threshold)
    sfm.fit(X, y)

    selected_feature_indices = sfm.get_support(indices=True)

    return [X.columns[i] for i in selected_feature_indices]
