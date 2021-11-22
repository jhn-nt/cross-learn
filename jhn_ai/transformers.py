import numpy as np
import pandas as pd

from . import _utils as ut
from . import _types as tp
from ._controllers import validate_input


from imblearn.over_sampling import SMOTENC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


from typing import Union, List, Optional, Callable, Dict, Type, Any
from numpy.typing import ArrayLike, DTypeLike
from warnings import warn

Covariates = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Targets = Type[Union[ArrayLike, pd.Series, List[float]]]
Probabilites = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Groups = Type[Union[ArrayLike, DTypeLike, pd.Series, List[Any]]]


class ColinearityRemover(BaseEstimator, TransformerMixin):
    """
    Removes linearly correlated features.


    Parameters
    ----------
    threshold : float, optional
        Threshold below which features are discarded.
        The default is 0.95.
    method : str, optional
        Method used to compute correlations,
        must be in {'pearson','spearman','kendall'}.
        The default is "pearson".
    """

    def __init__(self, threshold: float = 0.95, method: str = "pearson"):

        self.threshold = threshold
        self.method = method

    def fit(self, X: Covariates, y: Optional[Targets] = None):
        _ = validate_input(X, y, None, ignore_y=True)

        if isinstance(X, np.ndarray):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        r = _X.corr(method=self.method).abs()
        bool_mask = np.ones(r.shape)
        upper_r = r.where(np.triu(bool_mask, 1).astype(bool))
        to_drop = [
            column
            for column in upper_r.columns
            if any(upper_r[column] > self.threshold)
        ]
        to_keep = set(_X.columns.to_list()).difference(to_drop)

        self.colinear_features_ = to_drop
        self.selected_features_ = list(to_keep)

        # computes support mask
        self._support = []
        for col in _X.columns.to_list():
            if col in self.selected_features_:
                self._support.append(True)
            else:
                self._support.append(False)

        return self

    def transform(self, X: Covariates, y: Optional[Targets] = None):
        _ = validate_input(X, y, None, ignore_y=True)

        if isinstance(X, np.ndarray):
            _X = pd.DataFrame(X)
        else:
            _X = X.copy()

        X_transformed = _X.drop(self.colinear_features_, axis=1)
        return X_transformed

    def get_support(self):
        return np.array(self._support)


class IsolationForestV2(IsolationForest):
    """
    sklearn IsolationForest made compatible with imblearn Pipelines.

    Extends IsolationForest compatibility for outliers removal within pipelines.
    ALPHA
    """

    def fit_resample(self, X: Covariates, y: Targets):
        _ = validate_input(X, y, None, ignore_y=False, ignore_X=False)
        inliners = super().fit_predict(X, y=None)
        inliners_ix = np.where(inliners == 1)[0]

        X_sampled = ut._index_X(X, inliners_ix)
        if y is not None:
            y_sampled = ut._index_y(y, inliners_ix)
        else:
            y_sampled = None
        return X_sampled, y_sampled


class RateImputer(SimpleImputer):
    """
    Imputation transformer for handling missing values.

    Revision of the vanilla sklearn SimpleImputer but with:
        - get_support: method to track dropped features.
        - max_na_rate: maximum rate of missing values allowed for feature.



    Parameters
    ----------
    max_na_rate : float, optional
        Float indicating the maximum consented rate of missing
        values for each column. Columns with missing rates greater
        than max_na_rate are dropped.
        The default is 1.
    **kwargs : TYPE
        All other arguments supported by sklearn SimpleImputer.


    """

    def __init__(
        self,
        missing_values: Optional[Union[int, float, str]] = np.nan,
        strategy: str = "mean",
        fill_value: Optional[Union[int, float, str]] = None,
        max_na_rate: float = 1,
        verbose: int = 0,
        copy: bool = True,
        add_indicator: bool = False,
    ):
        warn("RateImputer will be deprecated")
        _ = super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )
        self.max_na_rate = max_na_rate

    def fit(
        self,
        X: Covariates,
        y: Optional[Targets] = None,
        groups: Optional[Groups] = None,
    ):
        _ = validate_input(X, y, groups, ignore_y=True, ignore_groups=True)
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X

        self.support = np.isnan(X_array).mean(axis=0) < self.max_na_rate
        X_filtered = X_array[:, self.support]
        _ = super().fit(X_filtered, y)
        return self

    def transform(
        self,
        X: Covariates,
        y: Optional[Targets] = None,
        groups: Optional[Groups] = None,
    ):
        _ = validate_input(X, y, groups, ignore_y=True, ignore_groups=True)
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, np.ndarray):
            X_array = X

        X_filtered = X_array[:, self.support]
        X_transformed = super().transform(X_filtered)
        return X_transformed

    def get_support(self):
        return self.support


class AutoSMOTENC(SMOTENC):
    """
    SMOTENC with support for automatic categorical features identification.

    ALPHA
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):

        super().__init__(
            categorical_features=[0],
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X: Covariates,
        y: Optional[Targets] = None,
        groups: Optional[Groups] = None,
    ):
        _ = validate_input(X, y, groups, ignore_y=True, ignore_groups=True)
        if isinstance(X, pd.DataFRame):
            _X = X.copy()
        else:
            _X = pd.DataFrame(X)

        uni_X = X.nunique()
        idx = [ix for (ix, val) in enumerate(uni_X) if val <= 3]
        super().set_params(categorical_features=idx)
        super().fit(X, y=y, groups=groups)
        return self
