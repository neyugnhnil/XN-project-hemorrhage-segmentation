#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None

def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return None

# macro auc
def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    values: list[float] = []
    for i in range(y_true.shape[1]):
        value = safe_auc(y_true[:, i], y_prob[:, i])
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else 0.0

# macro aucprc
def macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    values: list[float] = []
    for i in range(y_true.shape[1]):
        value = safe_auprc(y_true[:, i], y_prob[:, i])
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else 0.0

# macro F1
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

# per label AUC
def per_label_auc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for i, label in enumerate(class_names):
        metrics[label] = safe_auc(y_true[:, i], y_prob[:, i])
    return metrics

# per label auprc
def per_label_auprc(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str])\
     -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for i, label in enumerate(class_names):
        metrics[label] = safe_auprc(y_true[:, i], y_prob[:, i])
    return metrics

# per label f1
def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for i, label in enumerate(class_names):
        metrics[label] = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
    return metrics

# convert probabilities into binary predictions using a vector of per-label thresholds
def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be a 1D array")
    if thresholds.shape[0] != y_prob.shape[1]:
        raise ValueError("threshold count must match number of labels")
    return (y_prob >= thresholds[None, :]).astype(np.int32)

# dice score
def dice_score(y_true_seg: np.ndarray, y_pred_seg: np.ndarray, threshold: float, eps: float) -> float:
    y_true_bin = (y_true_seg >= 0.5).astype(np.float32)
    y_pred_bin = (y_pred_seg >= threshold).astype(np.float32)
    intersection = float(np.sum(y_true_bin * y_pred_bin))
    denom = float(np.sum(y_true_bin) + np.sum(y_pred_bin))
    return float((2.0 * intersection + eps) / (denom + eps))

# mean dice score
def mean_dice_score(y_true_seg: np.ndarray, y_pred_seg: np.ndarray, threshold: float, eps: float = 1.0e-7) -> float:
    if y_true_seg.shape[0] == 0:
        return 0.0
    values = [
        dice_score(y_true_seg[i], y_pred_seg[i], threshold=threshold, eps=eps)
        for i in range(y_true_seg.shape[0])
        ]
    return float(np.mean(values))

# summary dict
def summarize_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str]
    ) -> dict[str, object]:

    return {
        "macro_auc": macro_auc(y_true, y_prob),
        "macro_auprc": macro_auprc(y_true, y_prob),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_label_auc": per_label_auc(y_true, y_prob, class_names),
        "per_label_auprc": per_label_auprc(y_true, y_prob, class_names),
        "per_label_f1": per_label_f1(y_true, y_pred, class_names)
        }