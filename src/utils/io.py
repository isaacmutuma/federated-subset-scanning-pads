"""
io.py
-----
Save and load helpers for model checkpoints and results.
"""

import os
import json
import torch
import numpy as np


def save_checkpoint(model, fold_idx, model_name, checkpoint_dir='models/checkpoints'):
    """
    Save model state dict to checkpoints directory.

    Parameters
    ----------
    model          : torch.nn.Module
    fold_idx       : int — fold index (0-based)
    model_name     : str — e.g. 'inception', 'autoencoder'
    checkpoint_dir : str — path to checkpoints folder
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir,
                        f'{model_name}_fold{fold_idx + 1}_best.pt')
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model, fold_idx, model_name,
                     checkpoint_dir='models/checkpoints'):
    """
    Load model state dict from checkpoints directory.

    Parameters
    ----------
    model          : torch.nn.Module — instantiated model with correct architecture
    fold_idx       : int
    model_name     : str
    checkpoint_dir : str

    Returns
    -------
    model with loaded weights
    """
    path = os.path.join(checkpoint_dir,
                        f'{model_name}_fold{fold_idx + 1}_best.pt')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def save_metrics(metrics_list, filename, results_dir='results/metrics'):
    """
    Save fold metrics list to a JSON file.

    Parameters
    ----------
    metrics_list : list of dicts
    filename     : str — e.g. 'inception_cv_results.json'
    results_dir  : str
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved to {path}")


def load_metrics(filename, results_dir='results/metrics'):
    """Load fold metrics from a JSON file."""
    path = os.path.join(results_dir, filename)
    with open(path) as f:
        return json.load(f)


def save_activations(activations, fold_idx, layer_name,
                      activations_dir='results/activations'):
    """
    Save activation arrays for a fold and layer.

    Parameters
    ----------
    activations     : np.ndarray
    fold_idx        : int
    layer_name      : str — e.g. 'enc_conv1'
    activations_dir : str
    """
    os.makedirs(activations_dir, exist_ok=True)
    path = os.path.join(activations_dir,
                        f'fold{fold_idx + 1}_{layer_name}.npy')
    np.save(path, activations)
    return path


def load_activations(fold_idx, layer_name,
                      activations_dir='results/activations'):
    """Load saved activation array."""
    path = os.path.join(activations_dir,
                        f'fold{fold_idx + 1}_{layer_name}.npy')
    return np.load(path)
