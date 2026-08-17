import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score

def compute_metrics(y_true, y_scores, threshold=None):
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
    keys = ['auc', 'sensitivity', 'specificity', 'balanced_acc']
    result = {}
    for k in keys:
        values = [m[k] for m in fold_metrics]
        result[f'{k}_mean'] = round(float(np.mean(values)), 4)
        result[f'{k}_std']  = round(float(np.std(values)), 4)
    return result

def print_fold_results(fold_idx, metrics):
    print(f"Fold {fold_idx + 1} — AUC: {metrics['auc']:.4f}, Sens: {metrics['sensitivity']:.4f}, Spec: {metrics['specificity']:.4f}, BalAcc: {metrics['balanced_acc']:.4f}")

def print_summary(agg):
    print(f"\nMean AUC:          {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
    print(f"Mean Sensitivity:  {agg['sensitivity_mean']:.4f} ± {agg['sensitivity_std']:.4f}")
    print(f"Mean Specificity:  {agg['specificity_mean']:.4f} ± {agg['specificity_std']:.4f}")
    print(f"Mean Balanced Acc: {agg['balanced_acc_mean']:.4f} ± {agg['balanced_acc_std']:.4f}")
