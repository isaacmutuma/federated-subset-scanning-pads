"""
histograms.py
-------------
Federated per-neuron histogram references for DeepScan.

Protocol (one frozen encoder shared by all sites):
  1. Server   : make_global_edges()  — fixes bin edges once, from its own training HC activations,
                and broadcasts them to every site.
  2. Each site: site_counts()        — bins its local HC activations with the SHARED edges and
                sends only the integer count matrix (n_neurons, n_bins).
  3. Server   : merge_counts()       — sums site counts into the population reference. Because the
                edges are shared, the merge is exact: identical to binning the pooled data.
  4. Each site: pvalues_from_counts()— scores its own patients locally against the reference.

The two outer bins are open-ended (-inf / +inf), so test activations outside the server's range
fall into the extreme bins instead of being clipped to the boundary of an arbitrary local range.
"""

import numpy as np


def make_global_edges(ref_acts, n_bins):
    """
    Equal-width bin edges per neuron, with open-ended outer bins.

    Parameters
    ----------
    ref_acts : np.ndarray, shape (n_windows, n_neurons) — server-side HC activations
    n_bins   : int — number of bins (>= 2)

    Returns
    -------
    edges : np.ndarray, shape (n_neurons, n_bins + 1); edges[:, 0] = -inf, edges[:, -1] = +inf
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    lo, hi = ref_acts.min(axis=0), ref_acts.max(axis=0)
    hi = np.where(hi > lo, hi, lo + 1e-6)   # guard constant neurons
    edges = np.linspace(lo, hi, n_bins + 1, axis=1)
    edges[:, 0], edges[:, -1] = -np.inf, np.inf
    return edges


def bin_index(acts, edges):
    """Bin index (0 .. n_bins-1) of each activation, per neuron. acts: (n, n_neurons)."""
    n_neurons = edges.shape[0]
    idx = np.empty(acts.shape, dtype=np.int64)
    for j in range(n_neurons):
        idx[:, j] = np.searchsorted(edges[j, 1:-1], acts[:, j], side='right')
    return idx


def site_counts(acts, edges):
    """
    Integer histogram counts of one site's HC activations under the shared edges.
    This matrix is the ONLY thing a site transmits.

    Returns
    -------
    counts : np.ndarray, shape (n_neurons, n_bins), dtype int64
    """
    n_neurons, n_bins = edges.shape[0], edges.shape[1] - 1
    idx = bin_index(acts, edges)
    counts = np.zeros((n_neurons, n_bins), dtype=np.int64)
    for j in range(n_neurons):
        counts[j] = np.bincount(idx[:, j], minlength=n_bins)
    return counts


def merge_counts(count_list):
    """Server-side aggregation: element-wise sum of site count matrices."""
    return np.sum(np.stack(count_list, axis=0), axis=0)


def pvalues_from_counts(counts, edges, test_acts, direction_mask, alpha=1.0):
    """
    One-tailed empirical p-values of test activations against a histogram reference.

    Parameters
    ----------
    counts         : (n_neurons, n_bins) reference counts (local or merged)
    edges          : (n_neurons, n_bins + 1) shared edges
    test_acts      : (n_test, n_neurons)
    direction_mask : (n_neurons,) bool — True: high values anomalous (upper tail),
                     False: low values anomalous (lower tail).
                     None: two-tailed, p = min(1, 2 * min(upper, lower)) — needs no PD data
    alpha          : Laplace pseudocount added to every bin (1.0 = add-one smoothing)

    Returns
    -------
    p : (n_test, n_neurons) — P(X at least as extreme as the test bin | HC reference),
        bin-inclusive, so p >= alpha / (n_ref + alpha * n_bins) > 0.
    """
    n_bins = counts.shape[1]
    probs = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_bins)
    upper = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]   # P(bin >= b)
    lower = np.cumsum(probs, axis=1)                     # P(bin <= b)
    idx = bin_index(test_acts, edges)
    rows = np.arange(counts.shape[0])
    if direction_mask is None:
        p = 2.0 * np.minimum(upper[rows, idx], lower[rows, idx])
    else:
        p = np.where(direction_mask[None, :], upper[rows, idx], lower[rows, idx])
    return np.clip(p, 1e-12, 1.0)
