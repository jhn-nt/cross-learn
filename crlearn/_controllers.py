import numpy as np
import pandas as pd

from functools import partial

from typing import Type, Union, List, Any
from numpy.typing import ArrayLike, DTypeLike

Covariates = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Targets = Type[Union[ArrayLike, pd.Series, List[float]]]
Groups = Type[Union[ArrayLike, DTypeLike, pd.Series, List[Any]]]


def none_check(func):
    def wrapper(value, allow_na=False):
        if value is not None:
            return func(value)
        elif allow_na:
            return None
        else:
            raise ValueError(f"{func.__name__}: value cannot be None")

    return wrapper


@none_check
def validate_X(X: Covariates):
    if isinstance(X, pd.DataFrame):
        _X = X.values
    elif isinstance(X, list):
        _X = np.array(X)
    elif isinstance(X, np.ndarray):
        _X = X
    else:
        raise ValueError("X must be a 2d array-like object")

    if _X.ndim != 2:
        raise ValueError("X must have 2 dimesnions: (n_samples, n_features)")
    return _X


@none_check
def validate_y(y: Targets):
    if isinstance(y, pd.Series):
        _y = y.values
    elif isinstance(y, list):
        _y = np.array(y)
    elif isinstance(y, np.ndarray):
        _y = y
    else:
        raise ValueError("y must be a 1d array-like object")

    if _y.ndim != 1:
        raise ValueError("y must have 1 dimesnion: (n_samples,)")
    return _y


@none_check
def validate_groups(groups: Groups):
    if isinstance(groups, pd.Series):
        _groups = groups.values
    elif isinstance(groups, list):
        _groups = np.array(groups)
    elif isinstance(groups, np.ndarray):
        _groups = groups
    else:
        raise ValueError("groups must be a 1d array-like object")

    if _groups.ndim != 1:
        raise ValueError("groups must have 1 dimesnion: (n_samples,)")
    return _groups


def validate_input(
    X: Covariates,
    y: Targets,
    groups: Groups,
    ignore_X=False,
    ignore_y=False,
    ignore_groups=True,
):
    raw_inputs = {
        partial(validate_X, allow_na=ignore_X): X,
        partial(validate_y, allow_na=ignore_y): y,
        partial(validate_groups, allow_na=ignore_groups): groups,
    }

    inputs = [k(v) for (k, v) in raw_inputs.items()]
    inputs_shapes = set([_input.shape[0] for _input in inputs if _input is not None])
    if len(inputs_shapes) > 1:
        raise ValueError("Input mismatch, assert n_samples is equal for all inputs")
    return tuple(inputs)
