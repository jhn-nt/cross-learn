import numpy as np
from sklearn import metrics
from typing import Dict, List
from numpy.typing import ArrayLike


def support(y_true: ArrayLike) -> ArrayLike:
    """Returns the amount of samples in each class.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.

    Returns
    -------
    ArrayLike
        Array where each element is the amount of samples per each class.
    """
    _, support = np.unique(y_true, return_counts=True)
    return support


def class_balance(y_true: ArrayLike) -> ArrayLike:
    """Returns the relative size in percent of class.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.

    Returns
    -------
    ArrayLike
        Array where each element is the precent of samples per each class.
    """
    class_support = support(y_true)
    return class_support / np.sum(class_support)


def class_index(y_true: ArrayLike) -> ArrayLike:
    """Returns the class labels.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.

    Returns
    -------
    ArrayLike
        Class integer labels.
    """
    return np.unique(y_true)


def roc_curve_ovr(
    y_true: ArrayLike, y_score: ArrayLike, resolution: float = 0.01
) -> Dict[str, ArrayLike]:
    """Computes ROC curves.

    Extension of `sklearn.metrics.roc_curve` to the multi-class case.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.
    y_score : ArrayLike
        Probabilities of each sample belonging to a class.
    resolution : float, optional
        Resolution of the ROC curve, by default 0.01

    Returns
    -------
    Dict[str, ArrayLike]
        A dictionary with an `fpr` key containing the false positive rates.
        It contains a key for each class each containing the respective true positive rates.
    """

    def naive_ohe(y_true: ArrayLike) -> ArrayLike:
        ohe = np.zeros((y_true.shape[0], np.max(y_true) + 1))
        ohe[np.arange(y_true.shape[0]), y_true] = 1
        non_empty_classes = np.where(ohe.sum(axis=0))[0]
        return ohe[..., non_empty_classes]

    y_ohe = naive_ohe(y_true)
    interp_fpr = np.arange(0.0, 1.0 + resolution, resolution)
    rocs = {"fpr": interp_fpr}
    for i in range(y_ohe.shape[1]):
        fpr, tpr, _ = metrics.roc_curve(y_ohe[..., i], y_score[..., i])
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        rocs[i] = interp_tpr
    return rocs


def pr_curve_ovr(
    y_true: ArrayLike, y_score: ArrayLike, resolution: float = 0.01
) -> Dict[str, ArrayLike]:
    """Computes PR curves.

    Extension of `sklearn.metrics.precision_recall_curve` to the multi-class case.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.
    y_score : ArrayLike
        Probabilities of each sample belonging to a class.
    resolution : float, optional
        Resolution of the ROC curve, by default 0.01

    Returns
    -------
    Dict[str, ArrayLike]
        A dictionary with an `fpr` key containing the false positive rates.
        It contains a key for each class each containing the respective true positive rates.
    """

    def naive_ohe(y_true: ArrayLike) -> ArrayLike:
        ohe = np.zeros((y_true.shape[0], np.max(y_true) + 1))
        ohe[np.arange(y_true.shape[0]), y_true] = 1
        non_empty_classes = np.where(ohe.sum(axis=0))[0]
        return ohe[..., non_empty_classes]

    y_ohe = naive_ohe(y_true)
    interp_precision = np.arange(0.0, 1.0 + resolution, resolution)
    precision_recall = {"precision": interp_precision}
    for i in range(y_ohe.shape[1]):
        precision, recall, _ = metrics.precision_recall_curve(
            y_ohe[..., i], y_score[..., i]
        )
        interp_recall = np.interp(interp_precision, precision, recall)
        precision_recall[i] = interp_recall
    return precision_recall


def decision_curve_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, thresholds: ArrayLike
) -> ArrayLike:
    """Computes the net benefit at a given threshold.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.
    y_score : ArrayLike
        Probabilities of each sample belonging to a class.
    thresholds : ArrayLike
        Risk threshold where to evaluate the risk benefits.

    Returns
    -------
    ArrayLike
        Net benefit for each threshold in thresholds.
    """
    N = y_true.shape[0]

    def local_net_benefit(threshold):
        y_pred = np.where(y_score > threshold, 1, 0)
        tp = np.sum(np.logical_and(y_true, y_pred))
        fp = np.sum(np.where(y_pred > y_true, 1, 0))
        norm = threshold / (1 - threshold)
        return (tp - (fp * norm)) / N

    return np.vectorize(local_net_benefit)(thresholds)


def decision_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    resolution: float = 0.01,
    add_reference: bool = False,
) -> Dict[str, ArrayLike]:
    """Computes the net benefits for each class.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.
    y_score : ArrayLike
        Probabilities of each sample belonging to a class.
    resolution : float
        Desired resolution for the descion curves.
    add_reference: bool
        Whether to include a 'treat_all' reference column for each class, Default to False.

    Returns
    -------
    ArrayLike
        Net benefit for each class.
    """

    def naive_ohe(y_true):
        ohe = np.zeros((y_true.shape[0], np.max(y_true) + 1))
        ohe[np.arange(y_true.shape[0]), y_true] = 1
        non_empty_classes = np.where(ohe.sum(axis=0))[0]
        return ohe[..., non_empty_classes]

    y_ohe = naive_ohe(y_true)
    thresholds = np.arange(0.0, 1.0 - resolution, resolution)
    net_benefit_thresholds = {"threshold": thresholds}
    for i in range(y_ohe.shape[1]):
        net_benefit = decision_curve_at_threshold(
            y_ohe[..., i], y_score[..., i], thresholds
        )
        net_benefit_thresholds[i] = net_benefit

    if add_reference:
        reference_thresholds = decision_curve(y_true, np.ones_like(y_ohe))
        net_benefit_thresholds = {
            (key, ""): item for key, item in net_benefit_thresholds.items()
        }
        net_benefit_thresholds.update(
            {
                (key, "treat_all"): item
                for key, item in reference_thresholds.items()
                if key != "threshold"
            }
        )
    return net_benefit_thresholds


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, *args, **kwargs) -> ArrayLike:
    """Computes the area under the curve of the ROC.

    Extension of `sklearn.metrics.roc_auc_score` to support the binary case.

    Parameters
    ----------
    y_true : ArrayLike
        Targets.
    y_score : ArrayLike
        Probabilities of each sample belonging to a class.

    Returns
    -------
    ArrayLike
        AUCROC scores.
    """
    if y_score.shape[1] == 2:
        auc = metrics.roc_auc_score(y_true, y_score[..., 1], *args, **kwargs)
        output = np.array([auc, auc])
    else:
        output = metrics.roc_auc_score(y_true, y_score, *args, **kwargs)
    return output


def calibration_curve(
    y_true: ArrayLike, y_score: ArrayLike, resolution: float = 0.05
) -> Dict[str, ArrayLike]:
    """Computes calibration curves.

    Args:
        y_true (ArrayLike): Targets.
        y_score (ArrayLike): Probabilities of each sample belonging to a class.
        resolution (float, optional): Desired resolution for the calibration curves. Defaults to 0.05.

    Returns:
        Dict[str, ArrayLike]: Calibration curve for each class.
    """
    def naive_ohe(y_true):
        ohe = np.zeros((y_true.shape[0], np.max(y_true) + 1))
        ohe[np.arange(y_true.shape[0]), y_true] = 1
        non_empty_classes = np.where(ohe.sum(axis=0))[0]
        return ohe[..., non_empty_classes]

    y_ohe = naive_ohe(y_true)
    bins = np.arange(.0, 1.0, resolution)
    calibration_curves = {"mean_predicted_probability": bins}
    for i in range(y_ohe.shape[1]):
        digits = np.digitize(y_score[..., i], bins)
        expected = np.nan * np.zeros_like(bins)
        for j in np.unique(digits):
            expected[j-1] = np.mean(y_ohe[np.where(digits == j), i])
        calibration_curves[i] = expected
    return calibration_curves


def qq_curve(y_true:ArrayLike,y_pred:ArrayLike,n_thresholds:int=100,n_points:int=100)-> Dict[str, ArrayLike]:
    """Computes the Quantile-Quantile Plot between `y_true` and `y_score`.

    Args:
        y_true (ArrayLike): Targets.
        y_pred (ArrayLike): Predictions.
        n_thresholds (int, optional): Number of thresholds used to estimate quantiles. Defaults to 100.
        n_points (int, optional): Number of points used to interpolate quantiles. Defaults to 100.

    Returns:
        Dict[str, ArrayLike]: A dict with the thresholds for `y_true`,`y_pred` and the quantile associated to the threshold 
    """
    def ecdf(sample):
        q=np.quantile(sample,(.01,.99))
        thresholds=np.linspace(q[0],q[1],n_thresholds)
        densities=np.vectorize(lambda t: np.mean(sample<t))(thresholds)
        return densities, thresholds
    
    e=1/n_points
    q_pred=np.squeeze(np.interp(np.arange(e,1+e,e),*ecdf(y_pred)))
    q_true=np.squeeze(np.interp(np.arange(e,1+e,e),*ecdf(y_true)))

    return {"q_true":q_true,"q_pred":q_pred,"q":np.arange(e,1+e,e)}


def resilduals_curve(y_true:ArrayLike,y_pred:ArrayLike)-> Dict[str, ArrayLike]:
    """Computes regression residuals.

    Args:
        y_true (ArrayLike): Targets.
        y_pred (ArrayLike): Predictions.

    Returns:
        Dict[str, ArrayLike]: Residuals.
    """
    return {"predicted_value":np.asarray(y_pred),"residuals":np.asarray(y_true-y_pred)}


def exact_prediction_rate(y_true:ArrayLike,y_pred:ArrayLike,average:List[str]=None)->ArrayLike:
    raise NotImplementedError("Planned to be impelemented in the next release.")