"""
config.py
---------
Environment configuration for the federated-subset-scanning-pads project.
Edit ONLY this file when switching between environments.

Environments:
    - colab   : Google Colab + Google Drive
    - kaggle  : Kaggle Notebooks
    - local   : Local machine (CPU, for testing only)
"""

import os

# ── Set your environment here ─────────────────────────────────────────────────
ENV = 'colab'   # options: 'colab', 'kaggle', 'local'
# ─────────────────────────────────────────────────────────────────────────────

if ENV == 'colab':
    REPO_DIR     = '/content/federated-subset-scanning-pads'
    PADS_ROOT    = '/content/pads_dataset'          # extracted zip goes here
    PADS_ZIP     = '/content/drive/MyDrive/pads-parkisons-dataset-folder/pads-parkinsons-disease-smartwatch-dataset-1.0.0.zip'
    OUTPUT_DIR   = '/content/processed'             # windows.npy etc.
    CKPT_DIR     = '/content/drive/MyDrive/patchtst_checkpoints'   # survives disconnects
    RESULTS_DIR  = '/content/drive/MyDrive/patchtst_checkpoints'   # same folder

elif ENV == 'kaggle':
    REPO_DIR     = '/kaggle/working/federated-subset-scanning-pads'
    PADS_ROOT    = '/kaggle/input/datasets/isaacmu/pads-dataset/pads-parkinsons-disease-smartwatch-dataset-1.0.0'
    PADS_ZIP     = None                             # already extracted on Kaggle
    OUTPUT_DIR   = '/kaggle/working/processed'
    CKPT_DIR     = '/kaggle/working/checkpoints'
    RESULTS_DIR  = '/kaggle/working/results'

elif ENV == 'local':
    REPO_DIR     = os.path.expanduser('~/federated-subset-scanning-pads')
    PADS_ROOT    = os.path.expanduser('~/data/pads_dataset')
    PADS_ZIP     = None
    OUTPUT_DIR   = os.path.join(REPO_DIR, 'data', 'processed')
    CKPT_DIR     = os.path.join(REPO_DIR, 'models', 'checkpoints')
    RESULTS_DIR  = os.path.join(REPO_DIR, 'results', 'metrics')

else:
    raise ValueError(f"Unknown ENV: {ENV}. Must be 'colab', 'kaggle', or 'local'.")

# ── Model hyperparameters (never change these without updating memory) ─────────
D_MODEL      = 128
NUM_HEADS    = 8
NUM_LAYERS   = 3
FFN_DIM      = 256
DROPOUT      = 0.3
PATCH_LEN    = 16
STRIDE       = 8
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-2
T_MAX        = 50
PATIENCE     = 20
MAX_EPOCHS   = 100
N_SPLITS     = 5
RANDOM_STATE = 42
WINDOW_SIZE  = 200
N_WINDOWS    = 10
FS           = 100.0
LOWCUT       = 1.0
HIGHCUT      = 20.0
ROCKET_KERNELS = 1000

# ── Create directories ─────────────────────────────────────────────────────────
for d in [OUTPUT_DIR, CKPT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Config loaded — ENV={ENV}")
print(f"  PADS root:   {PADS_ROOT}")
print(f"  Checkpoints: {CKPT_DIR}")
print(f"  Results:     {RESULTS_DIR}")

TRAIN_WRIST = 'RightWrist'  # filter training data to right wrist only
