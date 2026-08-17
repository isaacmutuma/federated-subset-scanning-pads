"""
cnn_lstm.py
CNN-LSTM classifier for PADS IMU windows.

Input shape: (batch, n_channels, n_timesteps) = (batch, 6, 200)

Architecture:
  - CNN block: extracts spatial features from each time step across channels
  - LSTM block: captures temporal dependencies across the sequence
  - Classifier head: binary HC vs PD output

Reference: Fatimat01/PD-Detection 
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM classifier for 1D multivariate IMU time series.

    Parameters
    n_channels   : int — number of input channels (default 6: Accel XYZ + Gyro XYZ)
    n_timesteps  : int — number of time steps per window (default 200)
    n_classes    : int — number of output classes (default 2: HC, PD)
    cnn_filters  : list of int — filters per CNN layer
    kernel_size  : int — CNN kernel size
    lstm_hidden  : int — LSTM hidden units
    lstm_layers  : int — number of stacked LSTM layers
    dropout      : float — dropout rate
    """

    def __init__(self,
                 n_channels=6,
                 n_timesteps=200,
                 n_classes=2,
                 cnn_filters=(64, 128),
                 kernel_size=3,
                 lstm_hidden=128,
                 lstm_layers=2,
                 dropout=0.3):
        super().__init__()

        # CNN block 
        cnn_layers = []
        in_ch = n_channels
        for out_ch in cnn_filters:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output length after pooling
        cnn_out_len = n_timesteps
        for _ in cnn_filters:
            cnn_out_len = cnn_out_len // 2

        self.cnn_out_channels = cnn_filters[-1]
        self.cnn_out_len = cnn_out_len

        #  LSTM block 
        # CNN output: (batch, cnn_filters[-1], cnn_out_len)
        # LSTM expects: (batch, seq_len, features) → permute before passing
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Classifier head 
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """
        Parameters
     
        x : torch.Tensor, shape (batch, n_channels, n_timesteps)

        Returns
       
        logits : torch.Tensor, shape (batch, n_classes)
        """
        # CNN: (batch, n_channels, n_timesteps) → (batch, cnn_filters[-1], cnn_out_len)
        x = self.cnn(x)

        # Permute for LSTM: (batch, cnn_out_len, cnn_filters[-1])
        x = x.permute(0, 2, 1)

        # LSTM: (batch, seq_len, lstm_hidden)
        x, _ = self.lstm(x)

        # Take last time step
        x = x[:, -1, :]

        # Classify
        logits = self.classifier(x)
        return logits

    def get_activations(self, x, layer='lstm'):
        """
        Extract intermediate activations for a given layer.
        Used for DeepScan activation extraction.

        Parameters
        
        x     : torch.Tensor, shape (batch, n_channels, n_timesteps)
        layer : str — 'cnn_block1', 'cnn_block2', 'lstm', 'pre_classifier'

        Returns
 
        activations : torch.Tensor
        """
        activations = {}

        # Pass through CNN layers and record after each block
        out = x
        block_idx = 0
        for module in self.cnn:
            out = module(out)
            if isinstance(module, nn.MaxPool1d):
                activations[f'cnn_block{block_idx + 1}'] = out.detach()
                block_idx += 1

        # LSTM
        lstm_in = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        activations['lstm'] = lstm_out[:, -1, :].detach()

        # Pre-classifier
        pre_class = self.classifier[0](lstm_out[:, -1, :])
        pre_class = self.classifier[1](pre_class)
        pre_class = self.classifier[2](pre_class)
        activations['pre_classifier'] = pre_class.detach()

        if layer not in activations:
            raise ValueError(
                f"Layer '{layer}' not found. "
                f"Available: {list(activations.keys())}"
            )
        return activations[layer]


if __name__ == '__main__':
    # Quick check
    model = CNNLSTM(n_channels=6, n_timesteps=200, n_classes=2)
    x = torch.randn(8, 6, 200)
    logits = model(x)
    print(f"Input:   {x.shape}")
    print(f"Output:  {logits.shape}")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")

    # Test activation extraction
    for layer in ['cnn_block1', 'cnn_block2', 'lstm', 'pre_classifier']:
        act = model.get_activations(x, layer=layer)
        print(f"Layer '{layer}' activations: {act.shape}")