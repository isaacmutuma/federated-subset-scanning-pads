"""
dataset.py
----------
PyTorch Dataset class for PADS IMU windows.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PADSDataset(Dataset):
    """
    Dataset for PADS IMU windows.

    Parameters
    ----------
    windows     : np.ndarray, shape (n_windows, n_channels, n_samples)
    labels      : np.ndarray, shape (n_windows,) — 0=HC, 1=PD
    subject_ids : np.ndarray, shape (n_windows,) — subject ID per window
    """

    def __init__(self, windows, labels, subject_ids=None):
        self.windows     = torch.tensor(windows, dtype=torch.float32)
        self.labels      = torch.tensor(labels,  dtype=torch.long)
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class HCOnlyDataset(Dataset):
    """
    Dataset containing only HC windows — used for autoencoder training.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
              Should already be filtered to HC subjects only.
    """

    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
