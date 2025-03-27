from sklearn.base import BaseEstimator, TransformerMixin

class AmplifyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.factor