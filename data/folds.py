"""
folds.py
--------
5-fold stratified cross-validation splits at subject level.
Generates and saves fold splits so they are identical across all experiments.
"""

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold


def generate_fold_splits(subject_ids, subject_labels, n_splits=5,
                          random_state=42, val_fraction=0.2,
                          save_path=None):
    """
    Generate subject-level stratified 5-fold CV splits.
    Within each fold's training set, holds out val_fraction as validation.

    Parameters
    ----------
    subject_ids    : np.ndarray, shape (n_subjects,) — unique subject IDs
    subject_labels : np.ndarray, shape (n_subjects,) — 0=HC, 1=PD per subject
    n_splits       : int   — number of CV folds (default 5)
    random_state   : int   — random seed (default 42)
    val_fraction   : float — fraction of train subjects held out for val
    save_path      : str or None — if given, saves splits to this .pkl path

    Returns
    -------
    list of dicts, each with keys:
        'train_subjects' : np.ndarray
        'val_subjects'   : np.ndarray
        'test_subjects'  : np.ndarray
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)
    rng = np.random.RandomState(random_state)

    folds = []
    for train_val_idx, test_idx in skf.split(subject_ids, subject_labels):
        train_val_subjects = subject_ids[train_val_idx]
        train_val_labels   = subject_labels[train_val_idx]
        test_subjects      = subject_ids[test_idx]

        # Stratified split of train_val into train and val
        val_skf = StratifiedKFold(n_splits=int(1 / val_fraction),
                                   shuffle=True, random_state=random_state)
        train_idx_inner, val_idx_inner = next(
            val_skf.split(train_val_subjects, train_val_labels)
        )
        train_subjects = train_val_subjects[train_idx_inner]
        val_subjects   = train_val_subjects[val_idx_inner]

        folds.append({
            'train_subjects': train_subjects,
            'val_subjects':   val_subjects,
            'test_subjects':  test_subjects,
        })

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(folds, f)
        print(f"Fold splits saved to {save_path}")

    return folds


def load_fold_splits(path):
    """
    Load precomputed fold splits from a .pkl file.

    Parameters
    ----------
    path : str — path to saved fold splits pickle

    Returns
    -------
    list of dicts with 'train_subjects', 'val_subjects', 'test_subjects'
    """
    with open(path, 'rb') as f:
        folds = pickle.load(f)
    return folds


def get_fold_windows(windows, labels, subject_ids, fold):
    """
    Given a fold dict, return the window arrays for train, val, and test splits.

    Parameters
    ----------
    windows     : np.ndarray, shape (n_windows, n_channels, n_samples)
    labels      : np.ndarray, shape (n_windows,)
    subject_ids : np.ndarray, shape (n_windows,) — subject ID per window
    fold        : dict with 'train_subjects', 'val_subjects', 'test_subjects'

    Returns
    -------
    train_windows, train_labels : HC training windows only
    val_windows,   val_labels   : HC validation windows only
    test_windows,  test_labels  : all test windows (HC + PD)
    """
    def mask_for(subjects):
        return np.isin(subject_ids, subjects)

    # Train — HC only
    train_mask = mask_for(fold['train_subjects'])
    train_hc   = train_mask & (labels == 0)
    train_windows = windows[train_hc]
    train_labels  = labels[train_hc]

    # Val — HC only
    val_mask = mask_for(fold['val_subjects'])
    val_hc   = val_mask & (labels == 0)
    val_windows = windows[val_hc]
    val_labels  = labels[val_hc]

    # Test — all subjects (HC + PD)
    test_mask    = mask_for(fold['test_subjects'])
    test_windows = windows[test_mask]
    test_labels  = labels[test_mask]

    return (train_windows, train_labels,
            val_windows,   val_labels,
            test_windows,  test_labels)
