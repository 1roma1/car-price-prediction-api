import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter=" "):
        self.delimiter = delimiter

    def fit(self, X, y=None):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        X = X.tolist()

        if isinstance(X[0], list):
            X = [
                [len(str(item).split(self.delimiter)) for item in row]
                for row in X
            ]
        else:
            X = [len(str(item).split(self.delimiter)) for item in X]

        return np.array(X, dtype=np.int32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names, dtype=object)
