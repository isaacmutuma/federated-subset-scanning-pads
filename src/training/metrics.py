import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score

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

def spec_at_sens(y_true, y_scores, target_sens=0.9):
    """Specificity at the highest threshold reaching target_sens (screening operating point)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return round(float(1.0 - fpr[np.argmax(tpr >= target_sens)]), 4)

def bootstrap_auc_ci(y_true, y_scores, n_boot=2000, alpha=0.05, seed=42):
    """Subject-level percentile bootstrap CI for AUC (resample subjects with replacement)."""
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    rng, aucs = np.random.RandomState(seed), []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'auc': round(float(roc_auc_score(y_true, y_scores)), 4),
            'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4)}

def paired_bootstrap_auc_diff(y_true, scores_a, scores_b, n_boot=2000, seed=42):
    """Paired bootstrap of AUC(a) - AUC(b) on the same subjects. Returns diff, 95% CI, p (two-sided)."""
    y_true, a, b = np.asarray(y_true), np.asarray(scores_a), np.asarray(scores_b)
    rng, diffs = np.random.RandomState(seed), []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], a[idx]) - roc_auc_score(y_true[idx], b[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {'diff': round(float(roc_auc_score(y_true, a) - roc_auc_score(y_true, b)), 4),
            'ci_low': round(float(np.percentile(diffs, 2.5)), 4),
            'ci_high': round(float(np.percentile(diffs, 97.5)), 4),
            'p': round(float(min(p, 1.0)), 4)}

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
