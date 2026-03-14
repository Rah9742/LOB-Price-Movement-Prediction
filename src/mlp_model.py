"""
MLP (Multi-Layer Perceptron) — FI-2010 LOB Dataset
====================================================
A fully connected feedforward neural network that takes
the flat 144-feature LOB vector as input and learns
non-linear relationships between features and the target
mid-price movement class.

The MLP sits logically between the linear models (Ridge,
Logistic) and the sequential model (LSTM) in the model
family. Unlike the linear models it can capture non-linear
feature interactions, but unlike the LSTM it has no notion
of time — each sample is treated independently as a flat
144-dimensional vector, exactly like Ridge, Logistic,
Random Forest and XGBoost.

Architecture
------------
Input   : (batch, 144)          — flat LOB feature vector
Linear  : 144  -> 256, ReLU, Dropout
Linear  : 256  -> 128, ReLU, Dropout
Linear  : 128  -> 64,  ReLU, Dropout
Output  : 64   -> 3             — logits for 3 classes

Device detection (automatic)
-----------------------------
The script automatically selects the best available device:
- Apple M1/M2 Mac : uses MPS (Metal Performance Shaders)
- NVIDIA GPU      : uses CUDA
- Everything else : uses CPU

To force a specific device, replace the auto-detection block
near the top of the script with one of the following:
    DEVICE = torch.device("mps")    # Apple Silicon Mac
    DEVICE = torch.device("cuda")   # NVIDIA GPU
    DEVICE = torch.device("cpu")    # CPU only

Evaluation
----------
- Folds 7, 8, 9 (matching all other models)
- Validation set = last 1/fold fraction of training data
- Early stopping on validation loss (patience = 10 epochs)
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds
- Trained model weights saved as mlp_fold{N}.pt per fold

Usage
-----
python mlp_model.py
python mlp_model.py ./data    # if data folder is elsewhere

Dependencies: numpy, torch, scikit-learn, data_loader, evaluator
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import load_folds
from src.evaluator import Evaluator

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR   = "data"
FOLDS      = [7, 8, 9]   # all 3 folds — matches all other models
HIDDEN     = [256, 128, 64]  # hidden layer sizes
BATCH_SIZE = 256          # larger batch = faster on MPS/CUDA
MAX_EPOCHS = 50           # maximum training epochs
PATIENCE   = 10           # early stopping patience — number of epochs
                          # without improvement before training stops
LR         = 0.001        # Adam optimiser learning rate
DROPOUT    = 0.3          # dropout rate between layers — slightly higher
                          # than LSTM since MLP has no recurrent regularisation
# -------------------------------------------------

# -------------------------------------------------
# DEVICE DETECTION
# Automatically selects the best available device.
#
# To force a specific device, comment out the block below
# and uncomment one of these lines instead:
#   DEVICE = torch.device("mps")    # Apple Silicon Mac
#   DEVICE = torch.device("cuda")   # NVIDIA GPU
#   DEVICE = torch.device("cpu")    # CPU only
# -------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Device: Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print("Device: CPU")


# ==================================================
# MLP MODEL DEFINITION
# ==================================================

class LOB_MLP(nn.Module):
    """
    Three-layer MLP for LOB mid-price movement prediction.

    Takes a flat 144-feature LOB snapshot and predicts whether
    the mid-price will go Up (1), stay Stationary (2), or go
    Down (3) over the next k=5 events.

    Each hidden layer uses BatchNorm for training stability,
    ReLU activation, and Dropout for regularisation.

    Parameters
    ----------
    input_size  : int   — number of input features (144)
    hidden      : list  — hidden layer sizes ([256, 128, 64])
    num_classes : int   — number of output classes (3)
    dropout     : float — dropout rate between layers (0.3)
    """

    def __init__(self,
                 input_size:  int   = 144,
                 hidden:      list  = None,
                 num_classes: int   = 3,
                 dropout:     float = DROPOUT):
        super(LOB_MLP, self).__init__()

        if hidden is None:
            hidden = HIDDEN

        layers = []
        in_size = input_size

        for h in hidden:
            layers.extend([
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),   # normalise activations per batch
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = h

        # Output layer — no activation, CrossEntropyLoss handles softmax
        layers.append(nn.Linear(in_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, 144)
        return self.network(x)     # shape: (batch, num_classes=3)


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_loaders(X_train: np.ndarray, y_train: np.ndarray,
                    X_val:   np.ndarray, y_val:   np.ndarray,
                    X_test:  np.ndarray, y_test:  np.ndarray,
                    batch_size: int = BATCH_SIZE):
    """
    Convert flat feature arrays into PyTorch DataLoaders.

    Unlike the LSTM, the MLP does not use sliding window sequences
    — each sample is an independent 144-feature vector. This keeps
    the MLP directly comparable to Ridge, Logistic, RF and XGBoost.

    Labels are remapped 1,2,3 -> 0,1,2 for PyTorch CrossEntropyLoss.
    They are mapped back to 1,2,3 after prediction.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels
    X_test,  y_test  : test features and labels
    batch_size       : DataLoader batch size (256)

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights_tensor
    """
    # Remap labels 1,2,3 -> 0,1,2 for PyTorch
    y_tr = y_train - 1
    y_va = y_val   - 1
    y_te = y_test  - 1

    # Compute class weights to handle class imbalance
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights_tensor = torch.tensor(
        weights, dtype=torch.float32
    ).to(DEVICE)

    # Convert to PyTorch tensors
    def to_tensors(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    train_ds = TensorDataset(*to_tensors(X_train, y_tr))
    val_ds   = TensorDataset(*to_tensors(X_val,   y_va))
    test_ds  = TensorDataset(*to_tensors(X_test,  y_te))

    # Shuffle training data each epoch, keep val/test in order
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"    Samples — train: {len(train_ds):,}  "
          f"val: {len(val_ds):,}  test: {len(test_ds):,}")

    return train_loader, val_loader, test_loader, class_weights_tensor


# ==================================================
# TRAINING FUNCTIONS
# ==================================================

def train_epoch(model, loader, criterion, optimiser):
    """
    Run one full pass through the training data.

    Parameters
    ----------
    model     : LOB_MLP
    loader    : DataLoader — training data
    criterion : loss function (CrossEntropyLoss with class weights)
    optimiser : Adam optimiser

    Returns
    -------
    avg_loss : float — average loss across all batches
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimiser.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate_epoch(model, loader, criterion):
    """
    Run one full pass through evaluation data (no gradient updates).

    Parameters
    ----------
    model     : LOB_MLP
    loader    : DataLoader — validation or test data
    criterion : loss function

    Returns
    -------
    avg_loss  : float
    all_preds : np.ndarray — predicted class indices (0,1,2)
    all_true  : np.ndarray — true class indices (0,1,2)
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_true   = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            preds   = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    return total_loss / len(loader.dataset), all_preds, all_true


def train_model(model, train_loader, val_loader,
                class_weights, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, lr=LR):
    """
    Full training loop with early stopping on validation loss.

    Trains for up to max_epochs epochs. If validation loss does
    not improve for `patience` consecutive epochs, training stops
    early and the best weights are restored.

    Parameters
    ----------
    model         : LOB_MLP
    train_loader  : DataLoader — training data
    val_loader    : DataLoader — validation data
    class_weights : torch.Tensor — per-class loss weights
    max_epochs    : int — maximum number of epochs (50)
    patience      : int — early stopping patience (10)
    lr            : float — Adam learning rate (0.001)

    Returns
    -------
    model   : LOB_MLP with best weights restored
    history : dict — train/val loss and val F1 per epoch
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0
    history        = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\n    Training for up to {max_epochs} epochs "
          f"(early stopping patience={patience})...")
    print(f"    {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'Val F1':>8}")
    print(f"    {'-'*6} {'-'*12} {'-'*10} {'-'*8}")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimiser)
        val_loss, val_preds, val_true = evaluate_epoch(
            model, val_loader, criterion
        )
        val_f1 = f1_score(val_true, val_preds, average="macro",
                          zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"    {epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{val_f1:>8.4f}", end="")

        # Save best weights and reset patience counter if improved
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in
                              model.state_dict().items()}
            patience_count = 0
            print("  ✓ best")
        else:
            patience_count += 1
            print(f"  (patience {patience_count}/{patience})")
            if patience_count >= patience:
                print(f"\n    Early stopping triggered at epoch {epoch}")
                break

    # Restore the weights from the best epoch
    model.load_state_dict(best_weights)
    print(f"    Best val loss: {best_val_loss:.4f}")

    return model, history


# ==================================================
# MAIN TRAINING LOOP
# ==================================================

def run(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  MLP — FI-2010 LOB")
    print("=" * 60 + "\n")

    folds     = load_folds(data_dir=data_dir, folds_to_use=FOLDS)
    evaluator = Evaluator()

    for fold_data in folds:
        fold             = fold_data["fold"]
        X_train, y_train = fold_data["train"]
        X_val,   y_val   = fold_data["val"]
        X_test,  y_test  = fold_data["test"]

        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        # --- Prepare DataLoaders ---
        train_loader, val_loader, test_loader, class_weights = \
            prepare_loaders(X_train, y_train, X_val, y_val,
                            X_test,  y_test,  BATCH_SIZE)

        # --- Build model and move to selected device ---
        model = LOB_MLP(
            input_size  = 144,
            hidden      = HIDDEN,
            num_classes = 3,
            dropout     = DROPOUT
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"\n    Model parameters: {n_params:,}")
        print(f"    Architecture: 144 -> "
              f"{' -> '.join(str(h) for h in HIDDEN)} -> 3")

        # --- Train with early stopping ---
        model, history = train_model(
            model, train_loader, val_loader, class_weights,
            max_epochs = MAX_EPOCHS,
            patience   = PATIENCE,
            lr         = LR
        )

        # --- Evaluate on test set ---
        print(f"\n    Evaluating on test set...")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        _, test_preds, test_true = evaluate_epoch(
            model, test_loader, criterion
        )

        # Remap labels back from 0,1,2 to 1,2,3
        test_preds = test_preds + 1
        test_true  = test_true  + 1

        # --- Record results via shared evaluator ---
        evaluator.record("MLP", fold, test_true, test_preds)

        # --- Save trained model weights ---
        # To reload later without retraining:
        #   model = LOB_MLP()
        #   model.load_state_dict(torch.load(f"mlp_fold{fold}.pt"))
        save_path = f"mlp_fold{fold}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"    Model weights saved -> {save_path}")

    # --- Final summary across all folds ---
    evaluator.summary("MLP")
    evaluator.save("results_mlp.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
