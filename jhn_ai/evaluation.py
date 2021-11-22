import pandas as pd
import numpy as np
import sklearn as sk
import imblearn as imsk

from . import _utils as ut
from ._types import Estimator, CrossValidator
from ._controllers import validate_input
from tqdm import tqdm


from sklearn.model_selection import KFold
from sklearn import metrics


from functools import partial

from typing import Union, List, Optional, Callable, Dict, Type, Any
from numpy.typing import ArrayLike, DTypeLike
from warnings import warn

Covariates = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Targets = Type[Union[ArrayLike, pd.Series, List[float]]]
Probabilites = Type[Union[ArrayLike, pd.DataFrame, List[List[float]]]]
Groups = Type[Union[ArrayLike, DTypeLike, pd.Series, List[Any]]]


def regression_scores(
    y_true: Targets,
    y_pred: Targets,
    custom_scoring: Optional[Dict[str, Callable]] = None,
    y_index: Optional[Union[pd.Series, DTypeLike]] = None,
) -> pd.Series:
    """
    Computes a comprhensive list of regression scores.

    Parameters
    ----------
    y_true : Targets
        Target values.
    y_pred : Targets
        Predicted values.
    custom_scoring : Optional[Dict[str, Callable]], optional
        Allows to include custom scoring functions.
        Please note that the callable should have signature equal to:
            func(y_true, y_pred, y_proba, y_index)
        where:
            - y_true: array like of ground truth labels.
            - y_pred: binarized predictions from the model, ie the output of predict.
            - y_proba: predicted probabilities, ie the output of predict_proba.
            - y_index: index accompaigning labels, useful to perform aggreagtions.
        The default is None.
    y_index : Optional[Union[pd.Series, DTypeLike]], optional
        Index accompaigning labels, useful to perform aggreagtions.
        The default is None.

    Returns
    -------
    pd.Series
        Series with scores.
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1
    assert len(y_pred) == len(y_true)

    try:
        _regression_scores = {
            "explained_variance": metrics.explained_variance_score,
            "max_error": metrics.max_error,
            "mean_absolute_error": metrics.mean_absolute_error,
            "mean_squared_error": metrics.mean_squared_error,
            "root_mean_squared_error": partial(
                metrics.mean_squared_error, squared=False
            ),
            "mean_squared_log_error": metrics.mean_squared_log_error,
            "median_absolute_error": metrics.median_absolute_error,
            "r2": metrics.r2_score,
            "mean_poisson_deviance": metrics.mean_poisson_deviance,
            "mean_gamma_deviance": metrics.mean_gamma_deviance,
            "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error,
        }
    except:
        warn(
            f"""sklearn version found {sk.__version__}, 
              consider updating to 0.24.2 or higher""",
            category=UserWarning,
        )
        _regression_scores = {
            "explained_variance": metrics.explained_variance_score,
            "max_error": metrics.max_error,
            "mean_absolute_error": metrics.mean_absolute_error,
            "mean_squared_error": metrics.mean_squared_error,
            "root_mean_squared_error": partial(
                metrics.mean_squared_error, squared=False
            ),
            "mean_squared_log_error": metrics.mean_squared_log_error,
            "median_absolute_error": metrics.median_absolute_error,
            "r2": metrics.r2_score,
            "mean_poisson_deviance": metrics.mean_poisson_deviance,
            "mean_gamma_deviance": metrics.mean_gamma_deviance,
        }

    res = {}
    for item in _regression_scores.items():
        try:
            res[item[0]] = item[1](y_true, y_pred)
        except:
            res[item[0]] = np.nan

    if custom_scoring is not None:
        for k, item in custom_scoring.items():
            try:
                res[k] = item(y_true, y_pred, None, y_index)
            except:
                res[k] = np.nan

    return pd.Series(res)


def classification_scores(
    y_true: Targets,
    y_pred: Targets,
    y_proba: Probabilites,
    custom_scoring: Optional[Dict[str, Callable]] = None,
    y_index: Optional[Union[pd.Series, DTypeLike]] = None,
):
    """
    Computes a comprhensive list of classification scores.

    Parameters
    ----------
    y_true : Targets
        Target values.
    y_pred : Targets
        Predicted values.
    y_proba : Probabilites
        Estimated probabilities for each class.
    custom_scoring : Optional[Dict[str, Callable]], optional
        Allows to include custom scoring functions.
        Please note that the callable should have signature equale to:
            func(y_true, y_pred, y_proba, y_index)
        where:
            - y_true: array like of ground truth labels.
            - y_pred: binarized predictions from the model, ie the output of predict.
            - y_proba: predicted probabilities, ie the output of predict_proba.
            - y_index: index accompaigning labels, useful to perform aggreagtions.
        The default is None.
    y_index : Optional[Union[pd.Series, DTypeLike]], optional
        Index accompaigning labels, useful to perform aggreagtions.
        The default is None.

    Returns
    -------
    pd.DataFrame
        Dataframe where each column represent a class.
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1
    assert len(y_pred) == len(y_true)

    _classification_scores_per_class = {
        "f1": partial(metrics.f1_score, average=None),
        "recall": partial(metrics.recall_score, average=None),
        "precision": partial(metrics.precision_score, average=None),
        "support": ut.support,
    }

    _classification_aucs_per_class = {
        "roc_auc_ovr": ut.auc_multiclass,
    }

    res = {}
    for item in _classification_scores_per_class.items():
        try:
            res[item[0]] = item[1](y_true, y_pred)
        except:
            res[item[0]] = np.nan

    for item in _classification_aucs_per_class.items():
        try:
            res[item[0]] = item[1](y_true, y_proba)
        except:
            res[item[0]] = np.nan

    if custom_scoring is not None:
        for k, item in custom_scoring.items():
            res[k] = item(y_true, y_pred, y_proba, y_index)

    scores_per_class = pd.DataFrame(res, index=np.unique(y_true))
    scores_per_class = scores_per_class.T

    return scores_per_class


def classification_metrics(
    estimator: Estimator,
    X: Covariates,
    y: Targets,
    fpr_interp: Union[List[float], ArrayLike] = np.arange(0, 1.05, 0.05),
    as_dataframe: bool = True,
    custom_scoring: Optional[Dict[str, Callable]] = None,
    y_index: Optional[Union[pd.Series, DTypeLike]] = None,
):
    """
    Given a trained classifier, computes all sklearn classification metrics.

    Lazy wrapper of classification scores.

    Parameters
    ----------
    estimator : Estimator
        sklearn-like classifier.
    X : Covariates
        Input values.
    y : Targets
        Target values.
    fpr_interp : Union[List[float], ArrayLike], optional
        ROC interpolation granularity. The default is np.arange(0, 1.05, 0.05).
    as_dataframe : bool, optional
        if True returns the output as a dataframe, otherwise a tuple. The default is True.
    custom_scoring : Optional[Dict[str, Callable]], optional
        Allows to include custom scoring functions.
        Please note that the callable should have signature equale to:
            func(y_true, y_pred, y_proba, y_index)
        where:
            - y_true: array like of ground truth labels.
            - y_pred: binarized predictions from the model, ie the output of predict.
            - y_proba: predicted probabilities, ie the output of predict_proba.
            - y_index: index accompaigning labels, useful to perform aggreagtions.
        The default is None.
    y_index : Optional[Union[pd.Series, DTypeLike]], optional
        Index accompaigning labels, useful to perform aggreagtions.
        The default is None.

    Returns
    -------
    scores : Union[pd.DataFrame, tuple]
        DESCRIPTION.
    roc : pd.DataFrame
        DESCRIPTION.
    """

    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)

    _y = ut.squeeze_targets_if_needed(y)
    _y_pred = ut.squeeze_targets_if_needed(y_pred)

    scores = classification_scores(
        _y, _y_pred, y_prob, custom_scoring=custom_scoring, y_index=y_index
    )
    roc = ut.compute_roc(y, y_prob, fpr_interp=fpr_interp)

    if as_dataframe:
        pass
    else:
        scores = tuple(scores.values)
    return scores, y_pred, roc


# METRIC
def regression_metrics(
    estimator: Estimator,
    X: Covariates,
    y: Targets,
    as_dataframe: bool = True,
    custom_scoring: Optional[Dict[str, Callable]] = None,
    y_index: Optional[Union[pd.Series, DTypeLike]] = None,
):
    """
    Given a trained regressor, computes all sklearn regression metrics.

    Lazy wrapper of regression scores.

    Parameters
    ----------
    estimator : Estimator
        sklearn-like classifier.
    X : Covariates
        Input values.
    y : Targets
        Target values.
    as_dataframe : bool, optional
        if True returns the output as a dataframe, otherwise a tuple. The default is True.
    custom_scoring : Optional[Dict[str, Callable]], optional
        Allows to include custom scoring functions.
        Please note that the callable should have signature equale to:
            func(y_true, y_pred, y_proba, y_index)
        where:
            - y_true: array like of ground truth labels.
            - y_pred: binarized predictions from the model, ie the output of predict.
            - y_proba: predicted probabilities, ie the output of predict_proba.
            - y_index: index accompaigning labels, useful to perform aggreagtions.
        The default is None.
    y_index : Optional[Union[pd.Series, DTypeLike]], optional
        Index accompaigning labels, useful to perform aggreagtions.
        The default is None.

    Returns
    -------
    scores : TYPE
        DESCRIPTION.

    """
    # to-do: add support for lorenz curves
    y_pred = estimator.predict(X)

    _y = ut.squeeze_targets_if_needed(y)
    _y_pred = ut.squeeze_targets_if_needed(y_pred)

    scores = regression_scores(
        _y, _y_pred, custom_scoring=custom_scoring, y_index=y_index
    )
    if as_dataframe:
        pass
    else:
        scores = tuple(scores.values)
    return scores, y_pred, None


@ut.suppress_all_warnings
def supervised_crossvalidation(
    estimator: Estimator,
    X: Covariates,
    y: Targets,
    groups: Optional[Groups] = None,
    cv: CrossValidator = KFold(),
    paradigm: str = "auto",
    collect_func: Optional[Callable] = None,
    custom_scoring: Optional[Dict[str, Callable]] = None,
    inner_groups: Optional[Groups] = None,
    fit_kwargs: dict = {},
    verbose: bool = True,
    fpr_interp: Union[ArrayLike, List[float]] = np.arange(0, 1.05, 0.05),
    disable_progressbar: bool = False,
):
    """
    Returns most sklearn-supported estimator evaluation metrics using crossvalidation.

    A highlevel flexible method to easily evaluate complex sklearn-supported models.

    This method aims to aggregate all sparse and fragmented crossvalidation functions
    in sklearn in one single wrapper.
    It doesn not yet suppot non iid samples and aggregation scores (ie rank).

    Parameters
    ----------
    estimator : Estimator
        sklearn-like estimator.
    X : Covariates
        Input values.
    y : Targets
        Target values.
    groups : Optional[Groups], optional
        Group labels for the samples used while splitting the dataset into train/test set.
        Assigned both in the inner and outer loop when inner_groups is None.
        The default is None.
    cv : CrossValidator, optional
        Sklearn like CV itearator. Any object supporting the method split(X, y, groups).
        The default is KFold().
    paradigm : str, optional
        Paradigm of the problem, must be either {'auto','classification','regression'}.
        If 'auto', automatically determines the nature of the problem.
        The default is "auto".
    collect_func : Optional[Callable], optional
        Callable inspecting some property of the estimator after training.
        The signature of the function should be func(estimator), where the estimator
        is intrafold-fitted.
        The default is None.
    custom_scoring : Optional[Dict[str, Callable]], optional
        By default supervised_crossvalidation uses the default metric from
        classification_scores when paradigm is classification and regression_scores
        when regression. To add custom scores pass a dictionsary of callables.
        Please note that the callable should have signature equal to:
            func(y_true, y_pred, y_proba, y_index)
        where:
            - y_true: array like of ground truth labels.
            - y_pred: binarized predictions from the model, ie the output of predict.
            - y_proba: predicted probabilities, ie the output of predict_proba.
            - y_index: index accompaigning labels, useful to perform aggreagtions.
        The default is None.
    inner_groups : Optional[Groups], optional
        Group labels for the samples used while splitting the dataset into train/test set
        during nested crossvalidation. inner_groups allows for complex stacked crossvalidation
        strategies.If None, inner_groups is set equal to groups.
        The default is None.
    fit_kwargs : dict, optional
        Accessory arguments to pass to the fit method. Example cases are passing
        weights of samples, earlystopping in LGBMClassifier and other non-strickly
        sklearn like classifiers.
        If estimator is a pipeline, fit params should be preceeded by
        the object name ie 'cl__sample_weights': [10, 15...10]
        The defauls is {}.
    verbose : bool, optional
        Does nothing, it will be deprecated.
        The default is True.
    fpr_interp : Union[ArrayLike, List[float]], optional
        If a classification problem, it defines the resolution of the ROCs.
        The default is np.arange(0, 1.05, 0.05).
    disable_progressbar : bool, optional
        If set to False shows progressbar as crossvalidation goes on.
        Default is False.
    suppress_warnings: bool, optional.
        Inherited from the decorator. Allows to suppress all warnings regardless.
        Default is True.

    Returns
    -------
    scores : pd.DataFrame
        Dataframe containing folds scores.
    rocs : pd.DataFrame
        TPR and FPR in each fold.
    predictions : pd.DataFrame
        Fold-wise predictions.
    cv_collect : list
        Fold wise outputs of collect_func, None if collect_func is None.
    fold_index_map : List[dict]
        List of dictonaries containing the index of samples for each fold.

    Raises
    ------
    paradigm must be regression, classification or auto
        Raised if anything different from "auto","classification" or "regression" is set as paradigm.
    """

    # to-do: for regression add lorenz-curves as equivalent roc curves in classification.
    # all input assertion should be done in an ad-hoc method
    _ = validate_input(X, y, groups, ignore_groups=True)
    _ = validate_input(X, y, inner_groups, ignore_groups=True)

    if inner_groups is None:
        # if inner_groups is empty, inherits groups
        inner_groups = groups

    # handling functions to use based on the nature of the problem.
    if paradigm == "auto":
        _paradigm = ut.determine_paradigm(y)
    else:
        _paradigm = paradigm

    if _paradigm == "classification":
        scoring_func = partial(classification_metrics, fpr_interp=fpr_interp)
        output_func = ut.crossvalidation_output_for_classification
    elif _paradigm == "regression":
        scoring_func = regression_metrics
        output_func = ut.crossvalidation_output_for_regression
    else:
        raise ValueError("paradigm must be regression, classification or auto")

    # preparing the classifier properties saver if requested.
    cv_collect = []

    # preparing all savers.
    train_trace = {"scores": [], "rocs": [], "predictions": []}
    test_trace = {"scores": [], "rocs": [], "predictions": []}
    fold_index_map = []
    n_splits = cv.get_n_splits(X=X, y=y, groups=groups)

    for fold, (train_ix, test_ix) in tqdm(
        enumerate(cv.split(X, y, groups=groups)),
        total=n_splits,
        disable=disable_progressbar,
    ):
        # splitting intrafold train and test set
        train_data = ut._index_X(X, train_ix), ut._index_y(y, train_ix)
        test_data = ut._index_X(X, test_ix), ut._index_y(y, test_ix)

        # splitting intrafold groups
        groups_train = ut._index_groups(inner_groups, train_ix)

        # inheriting original indexes
        train_index = ut._inherit_index(y, train_ix)
        test_index = ut._inherit_index(y, test_ix)
        fold_index_map.append({"fold": fold, "train": train_index, "test": test_index})

        if ut.hasparam(estimator.fit, "groups"):
            # some classifiers don't support the groups keyword during training
            _ = estimator.fit(*train_data, groups=groups_train, **fit_kwargs)
        else:
            _ = estimator.fit(*train_data, **fit_kwargs)

        if collect_func:
            # storing some classifier property if requested
            cv_collect.append(collect_func(estimator))

        for _set, _trace, _index in zip(
            [train_data, test_data],
            [train_trace, test_trace],
            [train_index, test_index],
        ):
            # collecting scores in the train and test set of the fold
            _scores, _predictions, _rocs = scoring_func(
                estimator, *_set, custom_scoring=custom_scoring, y_index=_index
            )
            _trace["scores"].append(_scores)
            _trace["rocs"].append(_rocs)
            _trace["predictions"].append([_set[1], _predictions])

    output = output_func(train_trace, test_trace, fold_index_map)
    # rocs is None if regression.
    scores, rocs, predictions = output

    return scores, rocs, predictions, cv_collect, fold_index_map


@ut.suppress_all_warnings
def find_inertia_elbow(
    estimator: Estimator,
    X: Covariates,
    max_clusters: int,
):
    """
    Given a Kmeans-based estimator and input data X, identifies the optimal
    number of clusters by the elbow of inertia method.

    Parameters
    ----------
    estimator : estimator object
        sklearn like estimator.
    X : Covariates
        Input data.
    max_clusters : int
        Maximum number to clusters to investigate.
    suppress_warnings: bool, optional.
        Inherited from the decorator. Allows to suppress all warnings regardless.
        Default is True.

    Returns
    -------
    k_opt : int
        Optimal number of clusters.
    scores : array like
        Inertia associated to each cluster size in k_schedule.
    k_schedule : array like
        List of cluster sizes tested.

    """

    if isinstance(estimator, sk.pipeline.Pipeline) | isinstance(
        estimator, imsk.pipeline.Pipeline
    ):
        is_pipeline = True
    else:
        is_pipeline = False

    scores = []
    k_schedule = np.arange(2, max_clusters, 1)
    for k in k_schedule:
        if is_pipeline:
            estimator[-1].set_params(n_clusters=k)
        else:
            estimator.set_params(n_clusters=k)

        _ = estimator.fit_predict(X)

        if is_pipeline:
            scores.append(estimator[-1].inertia_)
        else:
            scores.append(estimator.inertia_)
    scores = np.array(scores)
    k_opt = ut.elbow_triangle(k_schedule, scores)
    return k_opt, scores, k_schedule
