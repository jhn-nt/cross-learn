import numpy as np
from sklearn import metrics
from typing import Dict
from numpy.typing import ArrayLike


def support(y_true: ArrayLike) -> ArrayLike:
    _, support = np.unique(y_true, return_counts=True)
    return support


def class_index(y_true: ArrayLike) -> ArrayLike:
    return np.unique(y_true)


def roc_curve_ovr(
    y_true: ArrayLike, y_score: ArrayLike, resolution: float = 0.01
) -> Dict[str, ArrayLike]:
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
    N = y_true.shape[0]

    def local_net_benefit(threshold):
        y_pred = np.where(y_score > threshold, 1, 0)
        tp = np.sum(np.logical_and(y_true, y_pred))
        fp = np.sum(np.where(y_pred > y_true, 1, 0))
        norm = threshold / (1 - threshold)
        return (tp - (fp * norm)) / N

    return np.vectorize(local_net_benefit)(thresholds)


def decision_curve(
    y_true: ArrayLike, y_score: ArrayLike, resolution: float = 0.01
) -> Dict[str, ArrayLike]:
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
    return net_benefit_thresholds
