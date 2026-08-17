"""
metrics.py
----------
Evaluation metrics for binary classification and anomaly detection.
"""

import numpy as np
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              balanced_accuracy_score)


def compute_metrics(y_true, y_scores, threshold=None):
    """
    Compute AUC, sensitivity, specificity, and balanced accuracy.

    Parameters
    ----------
    y_true    : np.ndarray, shape (n_samples,) — binary labels 0=HC, 1=PD
    y_scores  : np.ndarray, shape (n_samples,) — anomaly scores (higher = more anomalous)
    threshold : float or None — if None, uses median of y_scores

    Returns
    -------
    dict with keys: auc, sensitivity, specificity, balanced_acc, threshold
    """
    auc = roc_auc_score(y_true, y_scores)

    if threshold is None:
        threshold = np.median(y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        'auc':          round(auc, 4),
        'sensitivity':  round(sensitivity, 4),
        'specificity':  round(specificity, 4),
        'balanced_acc': round(balanced_acc, 4),
        'threshold':    round(float(threshold), 6),
    }


def aggregate_fold_metrics(fold_metrics):
    """
    Compute mean and std across folds.

    Parameters
    ----------
    fold_metrics : list of dicts, each from compute_metrics()

    Returns
    -------
    dict with mean and std for each metric
    """
    keys = ['auc', 'sensitivity', 'specificity', 'balanced_acc']
    result = {}
    for k in keys:
        values = [m[k] for m in fold_metrics]
        result[f'{k}_mean'] = round(float(np.mean(values)), 4)
        result[f'{k}_std']  = round(float(np.std(values)), 4)
    return result


def print_fold_results(fold_idx, metrics):
    """Print a single fold result line."""
    print(
        f"Fold {fold_idx + 1} — "
        f"AUC: {metrics['auc']:.4f}, "
        f"Sens: {metrics['sensitivity']:.4f}, "
        f"Spec: {metrics['specificity']:.4f}, "
        f"BalAcc: {metrics['balanced_acc']:.4f}"
    )


def print_summary(agg):
    """Print mean ± std summary across folds."""
    print(f"\nMean AUC:          {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
    print(f"Mean Sensitivity:  {agg['sensitivity_mean']:.4f} ± {agg['sensitivity_std']:.4f}")
    print(f"Mean Specificity:  {agg['specificity_mean']:.4f} ± {agg['specificity_std']:.4f}")
    print(f"Mean Balanced Acc: {agg['balanced_acc_mean']:.4f} ± {agg['balanced_acc_std']:.4f}")
