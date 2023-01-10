from pathlib import Path

from joblib import load, dump
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


def select_features(X, y, threshold=0.01) -> list:
    path = Path('data/model/helpers/relevant_features.pkl')
    try:
        relevant_features = load(path)
        return relevant_features
    except:
        print('Generating relevant features...')

    model = Lasso()
    model.fit(X, y)

    sfm = SelectFromModel(model, threshold=threshold)
    sfm.fit(X, y)

    selected_feature_indices = sfm.get_support(indices=True)
    relevant_features = [X.columns[i] for i in selected_feature_indices]

    dump(relevant_features, path)
    return relevant_features
