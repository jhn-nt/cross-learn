import unittest


import numpy as np
import numpy.random as rng
from numpy.typing import ArrayLike


class Test(unittest.TestCase):
    @staticmethod
    def add_random_missing_values(X: ArrayLike, max_rate: float = 0.7) -> ArrayLike:
        rates = rng.uniform(0.0, max_rate, X.shape[1])
        n_samples = np.round(X.shape[0] * rates).astype("int")

        for i in range(X.shape[1]):
            missing_ix = rng.choice(np.arange(X.shape[0]), n_samples[i])
            X[missing_ix, i] = np.nan

        return X

    def test_binary_classification(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.datasets import make_classification

        from crlearn.transformers import DropColinCV, DropByMissingRateCV
        from crlearn.evaluation import crossvalidate_classification

        X, y = make_classification(
            n_samples=1000, n_features=100, n_redundant=25, n_classes=2
        )
        X = self.add_random_missing_values(X)

        model = Pipeline(
            [
                ("missing_filter", DropByMissingRateCV()),
                ("impiter", SimpleImputer()),
                ("linear_filter", DropColinCV()),
                ("classifier", LogisticRegression(random_state=0)),
            ]
        )

        _ = crossvalidate_classification(model, X, y, name="binary_classification_test")

    def test_mutliclass_classification(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.datasets import make_classification

        from crlearn.transformers import DropColinCV, DropByMissingRateCV
        from crlearn.evaluation import crossvalidate_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=100,
            n_redundant=25,
            n_classes=10,
            n_informative=50,
            n_clusters_per_class=2,
        )
        X = self.add_random_missing_values(X)

        model = Pipeline(
            [
                ("missing_filter", DropByMissingRateCV()),
                ("impiter", SimpleImputer()),
                ("linear_filter", DropColinCV()),
                ("classifier", LogisticRegression(random_state=0)),
            ]
        )

        _ = crossvalidate_classification(
            model, X, y, name="multiclass_classification_test"
        )

    def test_nested_mutliclass_classification(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.datasets import make_classification
        from sklearn.model_selection import RandomizedSearchCV

        from crlearn.transformers import DropColinCV, DropByMissingRateCV
        from crlearn.evaluation import crossvalidate_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=100,
            n_redundant=25,
            n_classes=10,
            n_informative=50,
            n_clusters_per_class=2,
        )
        X = self.add_random_missing_values(X)

        model = Pipeline(
            [
                ("missing_filter", DropByMissingRateCV()),
                ("impiter", SimpleImputer()),
                ("linear_filter", DropColinCV()),
                ("classifier", LogisticRegression(random_state=0)),
            ]
        )
        param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1]}

        optimizer = RandomizedSearchCV(model, param_grid, n_iter=4)

        _ = crossvalidate_classification(
            optimizer, X, y, name="nested_multiclass_classification_test"
        )

    def test_nested_regression(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.impute import SimpleImputer
        from sklearn.datasets import make_regression
        from sklearn.model_selection import RandomizedSearchCV

        from crlearn.transformers import DropColinCV, DropByMissingRateCV
        from crlearn.evaluation import crossvalidate_regression

        X, y = make_regression(
            n_samples=1000,
            n_features=100,
            n_informative=50,
        )
        X = self.add_random_missing_values(X)

        model = Pipeline(
            [
                ("missing_filter", DropByMissingRateCV()),
                ("impiter", SimpleImputer()),
                ("linear_filter", DropColinCV()),
                ("regressor", Ridge(random_state=0)),
            ]
        )
        param_grid = {"regressor__alpha": [0.001, 0.01, 0.1, 1]}

        optimizer = RandomizedSearchCV(model, param_grid, n_iter=4)

        _ = crossvalidate_regression(optimizer, X, y, name="nested_regression_test")
