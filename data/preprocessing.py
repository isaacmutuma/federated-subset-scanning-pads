"""
preprocessing.py
----------------
Signal preprocessing for PADS IMU windows.
All functions are stateless and operate on numpy arrays.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=1.0, highcut=20.0, fs=100.0, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to a signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
    lowcut  : float — lower frequency bound in Hz (default 1.0)
    highcut : float — upper frequency bound in Hz (default 20.0)
    fs      : float — sampling rate in Hz (default 100.0)
    order   : int   — filter order (default 4)

    Returns
    -------
    np.ndarray — filtered signal, same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.array([filtfilt(b, a, ch) for ch in signal])


def segment_windows(signal, window_size=200, step=200):
    """
    Segment a continuous signal into non-overlapping windows.

    Parameters
    ----------
    signal      : np.ndarray, shape (n_channels, n_samples)
    window_size : int — samples per window (default 200 = 2s at 100Hz)
    step        : int — step size between windows (default 200 = non-overlapping)

    Returns
    -------
    np.ndarray, shape (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = signal.shape
    starts = range(0, n_samples - window_size + 1, step)
    windows = np.array([signal[:, s:s + window_size] for s in starts])
    return windows


def compute_normalization_stats(windows):
    """
    Compute per-channel mean and std from a set of windows.
    Called on HC training windows only — stats then applied to val and test.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)

    Returns
    -------
    mean : np.ndarray, shape (n_channels,)
    std  : np.ndarray, shape (n_channels,)
    """
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    mean = flat.mean(axis=(0, 2))
    std  = flat.std(axis=(0, 2))
    std[std == 0] = 1.0  # avoid division by zero
    return mean, std


def apply_normalization(windows, mean, std):
    """
    Apply per-channel z-score normalization using precomputed stats.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
    mean    : np.ndarray, shape (n_channels,)
    std     : np.ndarray, shape (n_channels,)

    Returns
    -------
    np.ndarray — normalized windows, same shape as input
    """
    return (windows - mean[None, :, None]) / std[None, :, None]


def preprocess_recording(raw_signal, fs=100.0, window_size=200,
                          lowcut=1.0, highcut=20.0):
    """
    Full preprocessing pipeline for one subject recording.
    Applies bandpass filter then segments into windows.

    Parameters
    ----------
    raw_signal  : np.ndarray, shape (n_channels, n_samples)
    fs          : float — sampling rate
    window_size : int   — samples per window
    lowcut      : float — bandpass lower bound Hz
    highcut     : float — bandpass upper bound Hz

    Returns
    -------
    np.ndarray, shape (n_windows, n_channels, window_size)
    """
    filtered = bandpass_filter(raw_signal, lowcut=lowcut,
                                highcut=highcut, fs=fs)
    windows = segment_windows(filtered, window_size=window_size,
                               step=window_size)
    return windows
