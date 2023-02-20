import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    explained_variance_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from typing import Union, Dict, List, Callable
import warnings
from itertools import product
import warnings
from inspect import signature


def p_format(
    p_series: Union[list, np.ndarray, pd.Series], threshold: tuple = (0.001, "<.001")
):
    """
    Low-level helper to format pvalues before being published.

    Parameters
    ----------
    p_series : Union[list, np.ndarray, pd.Series]
        DESCRIPTION.
    threshold : tuple, optional
        DESCRIPTION. The default is (0.001, "<.001").

    Returns
    -------
    p_series : TYPE
        DESCRIPTION.

    """
    if type(p_series) != pd.Series:
        p_series = pd.Series(p_series)
    p_series = p_series.apply(
        lambda val: threshold[1] if val < threshold[0] else "{:.3f}".format(val)
    )
    return p_series


def q1(val: pd.Series):
    """
    Low-level helper to compute Q1.

    Parameters
    ----------
    val : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return val.quantile(0.25)


def q2(val: pd.Series):
    """
    Low-level helper to compute Q2.

    Parameters
    ----------
    val : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return val.quantile(0.5)


def q3(val: pd.Series):
    """
    Low-level helper to compute Q3.

    Parameters
    ----------
    val : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return val.quantile(0.75)


def padline(str1: str, str2: str, maxlen: int):
    """
    Low-level helper padding two strings to a fixed maxlen.

    Parameters
    ----------
    str1 : str
        DESCRIPTION.
    str2 : str
        DESCRIPTION.
    maxlen : int
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if maxlen <= len(str1) + len(str2):
        warnings.warn(
            "maxlen is shorter than string to pad, maxlen set to len(str1)+len(str2)"
        )

    padding = " " * (maxlen - len(str1) - len(str2))
    return str1 + padding + str2


def interpolate_roc(
    tpr: np.ndarray,
    fpr: np.ndarray,
    fpr_interp: Union[np.ndarray, list] = np.arange(0, 1.05, 0.05),
):
    """
    Low-level helper to extend ROCs granularity.

    Parameters
    ----------
    tpr : np.ndarray
        DESCRIPTION.
    fpr : np.ndarray
        DESCRIPTION.
    fpr_interp : Union[np.ndarray, list], optional
        DESCRIPTION. The default is np.arange(0, 1.05, 0.05).

    Returns
    -------
    fpr_interp : TYPE
        DESCRIPTION.
    tpr_interp : TYPE
        DESCRIPTION.

    """
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    return fpr_interp, tpr_interp


def compute_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_interp: np.ndarray = np.arange(0, 1.05, 0.05),
):
    """
    Compute ROCs from true levels and probabilities.

    Given ground truths, y_true, and estimated probabilities, computes label wise ROCs.

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_prob : np.ndarray
        DESCRIPTION.
    fpr_interp : np.ndarray, optional
        DESCRIPTION. The default is np.arange(0, 1.05, 0.05).

    Returns
    -------
    tpr_df : TYPE
        DESCRIPTION.

    """

    y_true_ohe = pd.get_dummies(y_true).values
    tpr_dict = {}
    for c in range(y_prob.shape[1]):
        try:
            fpr, tpr, _ = roc_curve(y_true_ohe[:, c], y_prob[:, c])
        except:
            fpr, tpr, _ = roc_curve(np.zeros(y_prob[:, c].shape), y_prob[:, c])

        fpri, tpri = interpolate_roc(tpr, fpr, fpr_interp=fpr_interp)
        tpr_dict[c] = tpri
        tpr_dict["fpr"] = fpri
    tpr_df = pd.DataFrame(tpr_dict).set_index("fpr")
    return tpr_df


def elbow_triangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Low-level helper used in the elbow method.

    Identify the elbow using the tringle method.


    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    y : np.ndarray
        DESCRIPTION.

    Returns
    -------
    x_opt : TYPE
        DESCRIPTION.

    """
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - x[0] * m
    y_triangle = m * x + b
    y_opt = y - y_triangle
    x_opt = x[np.argmin(y_opt)]
    return x_opt


def segment_df(df: pd.DataFrame):
    """
    Low-level helper for propensity score matching.

    Computes quantiles of the pscore and returns a dataframe with quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    _df : TYPE
        DESCRIPTION.

    """
    _cuts = pd.cut(
        df, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], include_lowest=True
    )
    _df = pd.concat([df, _cuts], axis=1)
    _df.columns = ["pscore", "q"]
    return _df


def explained_variance_mse_r2(y_true, y_pred):
    """
    Low-level helper for crossvalidate_regression.TO DEPRECATE

    handy wrapper computing R^2, MSE and explained varinace at once.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    v : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    """
    v = explained_variance_score(y_true, y_pred)
    m = mean_squared_error(y_true, y_pred)
    r = r2_score(y_true, y_pred)
    return v, m, r


def best_inner_cv_stats(hypsearch_obj):
    """
    Low-level helper for symmetric_nested_crossvalidation
    handy wrapper to extract wrngled results in cv jsons.TO DEPRECATE

    Parameters
    ----------
    hypsearch_obj : TYPE
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    bi = hypsearch_obj.best_index_
    cv_results = hypsearch_obj.cv_results_
    _scores = list(hypsearch_obj.scorer_.keys())
    _results = []
    for item in cv_results.items():
        for score, agg in product(_scores, ["mean", "std"]):
            if (score in item[0]) & (agg == item[0].split("_")[0]):
                _results.append([score, agg, item[1][bi]])

    results = pd.DataFrame(_results, columns=["metric", "stat", "std"])
    results = results.set_index(["metric", "stat"])
    params = hypsearch_obj.best_params_
    return results


def best_outer_cv_stats(hypsearch_obj, scores):
    """
    Low-level helper for symmetric_nested_crossvalidation
    handy wrapper to extract wrngled results in cv jsons.TO DEPRECATE

    Parameters
    ----------
    hypsearch_obj : TYPE
        DESCRIPTION.
    scores : TYPE
        DESCRIPTION.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    _scores = list(hypsearch_obj.scorer_.keys())
    _results = []
    for items, score in product(scores.items(), _scores):
        if score in items[0]:
            mean_score = np.mean(items[1])
            std_score = np.std(items[1])
            _results.append([score, "mean", mean_score])
            _results.append([score, "std", std_score])
    results = pd.DataFrame(_results, columns=["metric", "stat", "std"])
    results = results.set_index(["metric", "stat"])
    return results


def support(y_true: Union[np.ndarray, pd.Series], y_pred=None):
    """
    Computes a class support for classification problems.

    Parameters
    ----------
    y_true : Union[np.ndarray, pd.Series]
        DESCRIPTION.
    y_pred : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(y_true, np.ndarray):
        _y = pd.Series(y_true)
    else:
        _y = y_true
    return _y.value_counts().sort_index().to_list()


def auc_multiclass(y_true, y_prob):
    """
    Computes AUCs in multi-classess problem.

    Equivalent to sklearn.metrics.auc in the double class problem.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_prob : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    if isinstance(y_true, np.ndarray):
        _y = pd.Series(y_true)
        y_ohe = pd.get_dummies(_y)
    else:
        y_ohe = pd.get_dummies(y_true)

    res = []
    for i, current_class in enumerate(y_ohe.columns):
        auc = roc_auc_score(y_ohe[current_class], y_prob[:, i])
        res.append(auc)

    return res


def _index_X(X, idx):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    Returns
    -------
    X_sliced : TYPE
        DESCRIPTION.

    """
    if isinstance(X, pd.DataFrame):
        X_sliced = X.iloc[idx, :]
    else:
        X_sliced = X[idx, :]
    return X_sliced


def _index_y(y, idx):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(y, pd.Series):
        y_sliced = y.iloc[idx]
    elif isinstance(y, np.ndarray):
        y_sliced = y[idx]
    elif isinstance(y, pd.DataFrame):
        warnings.warn(
            "y-targets should be either a pd.Series or a np.ndarray, not a pd.DataFrame",
            category=UserWarning,
        )
        y_sliced = y.iloc[idx, :]

    return y_sliced


def _index_groups(group, idx):
    """
    Low-level helper for supervised_crossvalidation.
    Support multidimensional groups.

    Parameters
    ----------
    group : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    Returns
    -------
    group_sliced : TYPE
        DESCRIPTION.

    """

    if isinstance(group, pd.Series):
        group_sliced = group.iloc[idx]
    elif isinstance(group, np.ndarray):
        if group.ndim == 1:
            group_sliced = group[idx]
        elif group.ndim == 2:
            group_sliced = group[idx, :]
    elif group is None:
        group_sliced = None

    return group_sliced


def _inherit_index(obj, idx):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    obj : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    Returns
    -------
    ix : TYPE
        DESCRIPTION.

    """
    if isinstance(obj, pd.DataFrame):
        ix = obj.iloc[idx, :].index
    elif isinstance(obj, pd.Series):
        ix = obj.iloc[idx].index
    elif isinstance(obj, np.ndarray):
        ix = idx
    return ix


def _make_classification_scores_output(train_scores, test_scores, fim_df):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    train_scores : TYPE
        DESCRIPTION.
    test_scores : TYPE
        DESCRIPTION.
    fim_df : TYPE
        DESCRIPTION.

    Returns
    -------
    _scores : TYPE
        DESCRIPTION.

    """

    _metric_index = train_scores[0].index.to_list()
    _fold_index = fim_df.fold.to_list()

    _ts = pd.concat(test_scores, axis=0).reset_index(drop=True)
    _tr = pd.concat(train_scores, axis=0).reset_index(drop=True)

    _scores = pd.concat([_tr, _ts], axis=1)
    _scores.columns = pd.MultiIndex.from_product(
        [["train", "test"], _tr.columns.to_list()]
    )
    _scores.index = pd.MultiIndex.from_product([_fold_index, _metric_index])
    _scores.columns.names = ["set", "class"]
    _scores.index.names = ["fold", "metric"]
    return _scores


def _make_regression_scores_output(train_scores, test_scores, fim_df):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    train_scores : TYPE
        DESCRIPTION.
    test_scores : TYPE
        DESCRIPTION.
    fim_df : TYPE
        DESCRIPTION.

    Returns
    -------
    _scores : TYPE
        DESCRIPTION.

    """

    _metric_index = train_scores[0].index.to_list()
    _fold_index = fim_df.fold.to_list()

    _ts = pd.concat(test_scores, axis=0).reset_index(drop=True)
    _tr = pd.concat(train_scores, axis=0).reset_index(drop=True)

    _scores = pd.concat([_tr, _ts], axis=1)
    _scores.columns = ["train", "test"]
    _scores.index = pd.MultiIndex.from_product([_fold_index, _metric_index])
    _scores.columns.names = ["set"]
    _scores.index.names = ["fold", "metric"]
    return _scores


def _make_rocs_output(train_rocs, test_rocs, fim_df):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    train_rocs : TYPE
        DESCRIPTION.
    test_rocs : TYPE
        DESCRIPTION.
    fim_df : TYPE
        DESCRIPTION.

    Returns
    -------
    _scores : TYPE
        DESCRIPTION.

    """
    _metric_index = train_rocs[0].index.to_list()
    _fold_index = fim_df.fold.to_list()

    _ts = pd.concat(test_rocs, axis=0).reset_index(drop=True)
    _tr = pd.concat(train_rocs, axis=0).reset_index(drop=True)

    _scores = pd.concat([_tr, _ts], axis=1)
    _scores.columns = pd.MultiIndex.from_product(
        [["train", "test"], _tr.columns.to_list()]
    )
    _scores.index = pd.MultiIndex.from_product([_fold_index, _metric_index])
    _scores.columns.names = ["set", "class"]
    _scores.index.names = ["fold", "fpr"]
    return _scores


def _make_prediction_output(train_predictions, test_predictions, fim_df):
    """
    Low-level helper for supervised_crossvalidation.

    Parameters
    ----------
    train_predictions : TYPE
        DESCRIPTION.
    test_predictions : TYPE
        DESCRIPTION.
    fim_df : TYPE
        DESCRIPTION.

    Returns
    -------
    _preds : TYPE
        DESCRIPTION.

    """

    _fold_index = fim_df.fold.to_list()
    _preds_vect = []
    for test_preds, fold_ix in zip(test_predictions, fim_df.iterrows()):
        _true = return_values_as_ndarray(test_preds[0])[:, None]
        _pred = return_values_as_ndarray(test_preds[1])[:, None]
        _preds_nd = np.concatenate((_true, _pred), axis=1)
        _preds_vect.append(
            pd.DataFrame(
                _preds_nd, columns=["true", "prediction"], index=fold_ix[1].test
            )
        )
    _preds = pd.concat(_preds_vect, axis=0)

    return _preds


def crossvalidation_output_for_classification(train_trace, test_trace, fold_index_map):
    """
    Low-level helper for callable supervised_crossvalidation.

    It wraps the callables
    _make_prediction_output, _make_regression_scores_ouput and _make_prediction_output
    in one callable. This function should never be used for scopes different than the one
    above.


    Parameters
    ----------
    train_trace : TYPE
        DESCRIPTION.
    test_trace : TYPE
        DESCRIPTION.
    fold_index_map : TYPE
        DESCRIPTION.

    Returns
    -------
    scores : TYPE
        DESCRIPTION.
    rocs : TYPE
        DESCRIPTION.
    predictions : TYPE
        DESCRIPTION.

    """
    fim_df = pd.DataFrame(fold_index_map)
    scores = _make_classification_scores_output(
        train_trace["scores"], test_trace["scores"], fim_df
    )

    rocs = _make_rocs_output(train_trace["rocs"], test_trace["rocs"], fim_df)

    predictions = _make_prediction_output(
        train_trace["predictions"], test_trace["predictions"], fim_df
    )
    return scores, rocs, predictions


def crossvalidation_output_for_regression(train_trace, test_trace, fold_index_map):
    """
    Low-level helper for callable supervised_crossvalidation.

    It wraps the callables
    _make_prediction_output, _make_regression_scores_ouput and _make_prediction_output
    in one callable. This function should never be used for scopes different than the one
    above.

    Parameters
    ----------
    train_trace : TYPE
        DESCRIPTION.
    test_trace : TYPE
        DESCRIPTION.
    fold_index_map : TYPE
        DESCRIPTION.

    Returns
    -------
    scores : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    predictions : TYPE
        DESCRIPTION.

    """
    fim_df = pd.DataFrame(fold_index_map)
    scores = _make_regression_scores_output(
        train_trace["scores"], test_trace["scores"], fim_df
    )

    predictions = _make_prediction_output(
        train_trace["predictions"], test_trace["predictions"], fim_df
    )
    return scores, None, predictions


def determine_paradigm(y, threshold=10):
    """
    Low-level helper for callable supervised_crossavalidation.

    Given an outcome, it determines whether it  a regression or classification problem.
    It counts the unique values in y and ,if their higher than threshold (default 50),
    returns the string "regression", "classification" otherwise.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    threshold : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    paradigm : TYPE
        DESCRIPTION.

    """
    if np.unique(y).shape[0] > threshold:
        paradigm = "regression"
    else:
        paradigm = "classification"
    return paradigm


def return_values_as_ndarray(
    obj: Union[pd.Series, pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Low-level helper which given or a series or a dataframe or a ndarray,
    always returns an ndarray. To be used for compatibility reasons.


    Parameters
    ----------
    obj : Union[pd.Series, pd.DataFrame, np.ndarray]
        DESCRIPTION.

    Returns
    -------
    values : TYPE
        DESCRIPTION.

    """

    values = obj
    if isinstance(obj, pd.Series) | isinstance(obj, pd.DataFrame):
        values = obj.values
    return values


def squeeze_targets_if_needed(y: Union[pd.Series, np.ndarray]):
    """
    Low-level helper that tries to force the dimension of an array-like object to 1.

    It is intended to be used to targets before injestion in a sklearn like estimator.

    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        DESCRIPTION.

    Returns
    -------
    _y : TYPE
        DESCRIPTION.

    """
    if y.ndim > 1:
        warnings.warn(
            f"""Targets have ndim equal to {y.ndim} when they should have ndim equal to 1. targets will be squeezed""",
            category=UserWarning,
        )
        _y = np.squeeze(y)
    else:
        _y = y
    return _y


def findall(
    array: Union[np.ndarray, list, tuple, pd.Series],
    values: Union[np.ndarray, list, tuple, pd.Series],
) -> Dict[Union[float, int, str, tuple, list], np.ndarray]:
    """
    Returns the indexes of each element of values in array.

    This implementation is 100X faster than the older one.

    Parameters
    ----------
    array : Union[np.ndarray, list, tuple, pd.Series]
        array-like object.
    values : Union[np.ndarray, list, tuple, pd.Series]
        array-like object.

    Returns
    -------
    search_dict : Dict[Union[float, int, str, tuple, list], np.ndarray]
        Dictionary where each key is a value of values and the relative items represents
        all of the indexes of that value found in array.

    """

    _array = make_array_from_listlike(array)
    _values = make_array_from_listlike(values)

    search_dict = {}
    _ = [search_dict.update({v: np.where(_array == v)[0]}) for v in _values]
    return search_dict


def suppress_all_warnings(func, suppress_warnings=True):
    """
    Utility decorator suppressing all warnings.

    Parameters
    ----------
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    wrapper : TYPE
        DESCRIPTION.

    """

    def wrapper(*args, suppress_warnings=suppress_warnings, **kwargs):
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def make_array_from_listlike(
    series: Union[list, tuple, pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Converts a one dimensional array-like object in numpy aray.

    Parameters
    ----------
    series : Union[list, tuple, pd.Series, np.ndarray]
        One dimensional array-like object.

    Raises
    ------
    ValueError
        series must be a 1D and array-like tuple, list, pd.Series or ndarray.

    Returns
    -------
    _series : np.ndarray
        One dimensional np.array.

    """

    if isinstance(series, list):
        _series = np.array(series)
    elif isinstance(series, tuple):
        _series = np.array(series)
    elif isinstance(series, pd.Series):
        _series = series.values
    elif isinstance(series, np.ndarray):
        _series = series
    else:
        raise ValueError(
            "series must be a 1D and array-like tuple, list, pd.Series or ndarray."
        )

    return _series


def transpose_list(groups: List[list]) -> List[list]:
    """
    Transposing list of lists as if they were matrices.

    Example:
        groups = [[1, 2], [3, 4]]
        transposed_groups = transpose_list(groups)
        assert transposed_groups == [[1, 3], [2, 4]]

    Parameters
    ----------
    groups : List[list]
        DESCRIPTION.

    Returns
    -------
    List[list]
        DESCRIPTION.

    """
    return list(map(list, zip(*groups)))


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
