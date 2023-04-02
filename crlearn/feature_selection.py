import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp

from ._controllers import validate_input

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


from numpy.typing import ArrayLike
from typing import Tuple, Dict, Callable, Optional


class DropColin(BaseEstimator, TransformerMixin):
    _coef: ArrayLike

    def __init__(self, threshold: float = 0.9, method: str = "pearson"):
        """Removal of linearly correlated features.

        Parameters
        ----------
        threshold : float, optional
            Features with correlation index above threshold are discarded, by default .9
        method : str, optional
            Method to estimate correlations across features, can be one of ['pearson','spearman'], by default "pearson"
        """
        assert (
            threshold > 0.0 and threshold <= 1.0 and isinstance(threshold, float)
        ), "'threshold' must be a float in the interval (.0,1.0]"
        assert (
            method in self.catalogue().keys()
        ), f"'method' can only be on of:{list(self.catalogue().keys())}"
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)

        correlation_func = self.catalogue()[self.method]
        r = correlation_func(X)
        self.support = self.find_features_below_threshold(r, self.threshold)
        self.coef_ = r
        return self

    def transform(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)
        return X[:, self.get_support()]

    def get_support(self, indices=False):
        return np.where(self.support)[0] if indices else self.support

    @staticmethod
    def find_features_below_threshold(
        r: ArrayLike, threshold: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        r = np.abs(r)
        np.fill_diagonal(r, 0.0)
        mask = np.where(r > threshold, True, False)

        to_skip = np.ones((r.shape[0],), dtype="int")
        for i, local_mask in enumerate(mask):
            if to_skip[i] == 1:
                to_skip[np.where(local_mask)] = 0

        support = np.where(to_skip == 1, True, False)
        return support

    @staticmethod
    def pearson_coeff(X: ArrayLike) -> ArrayLike:
        covariance = np.cov(X, rowvar=False)
        sigma = np.expand_dims(np.std(X, axis=0), axis=1)
        sigma = np.matmul(sigma, sigma.T) + 1e-6
        return np.clip(covariance / sigma, -1.0, 1.0)

    @staticmethod
    def spearman_coeff(X: ArrayLike) -> ArrayLike:
        ranks = np.argsort(X, axis=0)
        return DropColin.pearson_coeff(ranks)

    @staticmethod
    def catalogue() -> Dict[str, Callable]:
        return {
            "pearson": DropColin.pearson_coeff,
            "spearman": DropColin.spearman_coeff,
        }


class DropColinCV(DropColin):
    coef_: ArrayLike

    def __init__(self, cv=KFold(), alpha=0.05, **kwargs):
        """Removal of linearly correlated features via crossvalidation.

        Parameters
        ----------
        threshold : float, optional
            Features with correlation index above threshold are discarded, by default .9
        method : str, optional
            Method to estimate correlations across features, can be one of ['pearson','spearman'], by default "pearson"
        cv : _type_, optional
            Crossvalidation strategy, by default KFold()
        alpha : float, optional
            Threshold of significance, by default .05
        """
        super(DropColinCV, self).__init__(**kwargs)
        self.cv = cv
        self.alpha = alpha

    def fit(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)
        self.coef_ = self.bootstrap_correlation_coefficients(X, y)
        self.support = self.find_features_significantly_below_threshold(
            self.coef_, self.threshold, self.alpha
        )
        return self

    def bootstrap_correlation_coefficients(
        self, X: ArrayLike, y: ArrayLike
    ) -> ArrayLike:
        support = []
        for infold, _ in self.cv.split(X, y):
            local_coef = super().fit(X[infold, :]).coef_
            np.fill_diagonal(local_coef, 0.0)
            support.append(np.expand_dims(local_coef, axis=-1))
        support = np.concatenate(support, axis=-1)
        return support

    @staticmethod
    def find_features_significantly_below_threshold(
        boostrapped_r: ArrayLike, threshold: float, alpha: float
    ) -> ArrayLike:
        boostrapped_r = np.abs(boostrapped_r)

        to_skip = np.ones((boostrapped_r.shape[0],), dtype="bool")
        for i, row in enumerate(boostrapped_r):
            if to_skip[i]:
                pval = ttest_1samp(
                    row, popmean=threshold, axis=1, alternative="less"
                ).pvalue
                to_skip[np.where(pval > alpha)] = False
        return to_skip


class DropByMissingRate(BaseEstimator, TransformerMixin):
    coef_: ArrayLike

    def __init__(self, max_na_rate: float = 0.3):
        """Drops features missing more than a pre-defined rate.

        Parameters
        ----------
        max_na_rate : float, optional
            Maximum rate of missing samples after which the feature is discarded, by default .3
        """
        self.max_na_rate = max_na_rate

    def fit(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)
        na_mask = pd.isna(X) # pandas supports isna for dtypes object out of the box.
        self.coef_ = np.mean(na_mask, axis=0)
        self.support = np.where(self.coef_ <= self.max_na_rate, True, False)
        return self

    def transform(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)
        return X[:, self.support]

    def get_support(self, indices=False):
        return np.where(self.support)[0] if indices else self.support


class DropByMissingRateCV(DropByMissingRate):
    ceof_: ArrayLike

    def __init__(self, alpha=0.05, cv=KFold(), **kwargs):
        """Drops features missing more than a pre-defined rate via cross-validation.

        Parameters
        ----------
        max_na_rate : float, optional
            Maximum rate of missing samples after which the feature is discarded, by default .3
        cv : _type_, optional
            Crossvalidation strategy, by default KFold()
        alpha : float, optional
            Threshold of significance, by default .05
        """
        super(DropByMissingRateCV, self).__init__(**kwargs)
        self.alpha = alpha
        self.cv = cv

    def fit(self, X, y=None):
        X, _, _ = validate_input(X, y, None, ignore_y=True)
        support = []
        for infold, _ in self.cv.split(X, y):
            na_mask = pd.isna(X[infold, :])
            local_rate = np.mean(na_mask, axis=0)
            support.append(local_rate)
        support = np.vstack(support)
        pvals = ttest_1samp(support, self.max_na_rate, alternative="less").pvalue
        self.coef_ = support
        self.support = np.where(pvals <= self.alpha, True, False)
        return self
