import pandas as pd
import numpy as np

from typing import Union, Dict, List, Callable, Optional, Any
from numpy.typing import ArrayLike
import warnings
import warnings
from inspect import signature




def elbow_triangle(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Low-level helper used in the elbow method.

    Identify the elbow using the tringle method.


    Parameters
    ----------
    x : ArraLike
    y : ArrayLike

    Returns
    -------
    x_opt : ArrayLike

    """
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - x[0] * m
    y_triangle = m * x + b
    y_opt = y - y_triangle
    x_opt = x[np.argmin(y_opt)]
    return x_opt



def index_X(X:ArrayLike, idx:ArrayLike)->ArrayLike:
    """Slices Input Data.

    Parameters
    ----------
    X : ArrayLike
        Input Data.
    idx : ArrayLike
        Indexes.

    Returns
    -------
    ArrayLike
        Indexed Input Data.
    """
    if hasattr(X,"iloc"):
        slice_f=getattr(X,"iloc")
    elif hasattr(X,"__getitem__"):
        slice_f=X
    return slice_f[idx,:]


def index_y(y:ArrayLike, idx:ArrayLike)->ArrayLike:
    """Slices Target Data.

    Parameters
    ----------
    y : ArrayLike
        Target Data.
    idx : ArrayLike
        Indexes.

    Returns
    -------
    ArrayLike
        Indexed Target Data.
    """
    if hasattr(y,"iloc"):
        slice_f=getattr(y,"iloc")
    elif hasattr(y,"__getitem__"):
        slice_f=y
    return slice_f[idx]


def index_groups(groups:Optional[ArrayLike], idx:ArrayLike)->Optional[ArrayLike]:
    """Slices Group Data.

    Parameters
    ----------
    group : Optional[ArrayLike]
        Group data.
    idx : ArrayLike
        Indexes.

    Returns
    -------
    Optional[ArrayLike]
        Indexed Group Data.
    """
    if groups is not None:
        if hasattr(groups,"iloc"):
            slice_f=getattr(groups,"iloc")
        elif hasattr(groups,"__getitem__"):
            slice_f=groups
        return slice_f[idx]
    else:
        return None






def suppress_all_warnings(func:Callable, suppress_warnings:bool=True)->Any:
    """
    Utility decorator suppressing all warnings.

    Parameters
    ----------
    func : Callable
        Input Method.
    suppress_warnings: bool.
        Flag to suppress the warnings, Defaults to True.

    Returns
    -------
    wrapper : Any
        Output of the warning suppressed method.

    """

    def wrapper(*args, suppress_warnings=suppress_warnings, **kwargs):
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def hasparam(func: Callable, param: str) -> bool:
    """
    Inspects signature of method func for parameter param.
    If param is in the signature returns True.
    Parameters
    ----------
    func : Callable
        Callable of which the signature is inspected.
    param : str
        param to search for in func signature.
    Returns
    -------
    bool
        Whether param is in the func signature.
    """
    return param in signature(func).parameters

