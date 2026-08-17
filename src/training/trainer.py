"""
trainer.py
----------
Generic training loop with early stopping and checkpointing.
Works for both classifier and autoencoder training.
"""

import torch
import numpy as np
from src.utils.io import save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
        else:
            # Autoencoder — reconstruction only
            x = batch.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Run evaluation. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
            else:
                x = batch.to(device)
                out = model(x)
                loss = criterion(out, x)
            total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                fold_idx, model_name, device,
                max_epochs=200, patience=20,
                checkpoint_dir='models/checkpoints',
                verbose=True):
    """
    Full training loop with early stopping and best-checkpoint saving.

    Parameters
    ----------
    model          : torch.nn.Module
    train_loader   : DataLoader
    val_loader     : DataLoader
    optimizer      : torch.optim optimizer
    criterion      : loss function
    fold_idx       : int — current fold (0-based)
    model_name     : str — used for checkpoint filename
    device         : torch.device
    max_epochs     : int
    patience       : int — early stopping patience
    checkpoint_dir : str
    verbose        : bool

    Returns
    -------
    model — loaded with best weights
    history — dict with 'train_loss' and 'val_loss' lists
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"Epoch {epoch}/{max_epochs}, "
                  f"Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, fold_idx, model_name, checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    # Load best checkpoint
    best_path = f"{checkpoint_dir}/{model_name}_fold{fold_idx + 1}_best.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    return model, history
