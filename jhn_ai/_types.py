from typing import Protocol
from typing import NewType


class Estimator(Protocol):
    def fit(self, X, y=None, **fit_params):
        pass

    def predict(self, X):
        pass

    def set_params(self, *args):
        pass

    def get_params(self, *args):
        pass


class CrossValidator(Protocol):
    def split(self, X, Y=None, groups=None):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        pass
