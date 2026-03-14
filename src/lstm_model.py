"""
LSTM — FI-2010 LOB Dataset
===========================
A Long Short-Term Memory (LSTM) recurrent neural network that
takes a sequence of 100 consecutive LOB snapshots as input and
learns temporal patterns across the sequence.

Unlike the static models (Ridge, Logistic, RF, XGBoost) which
treat each sample as an independent 144-feature vector, the LSTM
explicitly models how the order book evolves over time.

Architecture
------------
Input  : (batch, 100, 144)  — 100 timesteps, 144 features each
LSTM   : 1 layer, 64 hidden units
FC     : Dense(3) with softmax — maps to 3 classes

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
    DEVICE = torch.device("cpu")    # CPU only (slowest)

Evaluation
----------
- Folds 7, 8, 9 (matching all other models)
- Validation set = final training day, with prior-split context
  prepended for sequence evaluation
- Early stopping on validation loss (patience = 10 epochs)
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds
- Trained model weights saved as lstm_fold{N}.pt per fold

Usage
-----
python lstm_model.py
python lstm_model.py ./data    # if data folder is elsewhere

Dependencies: numpy, torch, scikit-learn, data_loader, evaluator
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import load_folds, make_sequences
from src.evaluator import Evaluator

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR    = "data"
FOLDS       = [7, 8, 9]   # all 3 folds — matches other models
WINDOW      = 100          # timesteps per sequence — matches DeepLOB standard
HIDDEN_SIZE = 64           # LSTM hidden units
NUM_LAYERS  = 1            # number of stacked LSTM layers
BATCH_SIZE  = 256          # larger batch = faster on MPS/CUDA
MAX_EPOCHS  = 50           # maximum training epochs
PATIENCE    = 10           # early stopping patience — number of epochs
                           # without improvement before training stops
LR          = 0.001        # Adam optimiser learning rate
DROPOUT     = 0.2          # dropout rate applied after LSTM layer
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
# LSTM MODEL DEFINITION
# ==================================================

class LOB_LSTM(nn.Module):
    """
    Single-layer LSTM for LOB mid-price movement prediction.

    Takes a sequence of LOB snapshots and predicts whether the
    mid-price will go Up (1), stay Stationary (2), or go Down (3)
    over the next k=5 events.

    Parameters
    ----------
    input_size  : int   — features per timestep (144)
    hidden_size : int   — LSTM hidden units (64)
    num_layers  : int   — stacked LSTM layers (1)
    num_classes : int   — output classes (3)
    dropout     : float — dropout rate after LSTM (0.2)
    """

    def __init__(self,
                 input_size:  int   = 144,
                 hidden_size: int   = HIDDEN_SIZE,
                 num_layers:  int   = NUM_LAYERS,
                 num_classes: int   = 3,
                 dropout:     float = DROPOUT):
        super(LOB_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,   # input shape: (batch, seq_len, features)
            dropout     = dropout if num_layers > 1 else 0.0
        )

        # Dropout applied to LSTM output before the fully connected layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer mapping hidden units to 3 classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len=100, input_size=144)
        lstm_out, _ = self.lstm(x)

        # Take only the output from the final timestep
        last_out = lstm_out[:, -1, :]    # shape: (batch, hidden_size)
        out      = self.dropout(last_out)
        out      = self.fc(out)          # shape: (batch, num_classes=3)
        return out


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_loaders(X_train: np.ndarray, y_train: np.ndarray,
                    X_val:   np.ndarray, y_val:   np.ndarray,
                    X_test:  np.ndarray, y_test:  np.ndarray,
                    window:     int = WINDOW,
                    batch_size: int = BATCH_SIZE):
    """
    Create sliding window sequences and wrap in PyTorch DataLoaders.

    Each sample becomes a (window=100, 144) sequence. The label
    is taken from the last timestep of each window.

    Labels are remapped 1,2,3 -> 0,1,2 for PyTorch CrossEntropyLoss.
    They are mapped back to 1,2,3 after prediction.

    The first (window-1) samples of the validation and test sets
    are used only to provide context for the first valid window —
    they are not included as prediction targets. This prevents any
    window from crossing the train/val or val/test boundary.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels
    X_test,  y_test  : test features and labels
    window           : number of timesteps per sequence (100)
    batch_size       : DataLoader batch size (256)

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights_tensor
    """
    def make_split_sequences(context_X: np.ndarray,
                             split_X: np.ndarray,
                             split_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build one window per target in split_X using preceding context."""
        context_len = window - 1
        if context_len <= 0:
            return split_X[:, None, :], split_y

        if len(context_X) < context_len:
            raise ValueError(
                f"Need at least {context_len} context rows, got {len(context_X)}"
            )

        combined_X = np.vstack([context_X[-context_len:], split_X])
        X_seq = np.zeros((len(split_X), window, split_X.shape[1]),
                         dtype=np.float32)

        for i in range(len(split_X)):
            X_seq[i] = combined_X[i:i + window]

        return X_seq, split_y.copy()

    # Train sequences do not have preceding context, so the first `window-1`
    # targets are unavailable. Validation/test use the prior split as context.
    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, window)
    X_va_seq, y_va_seq = make_split_sequences(X_train, X_val, y_val)
    X_te_seq, y_te_seq = make_split_sequences(X_val, X_test, y_test)

    # Remap labels 1,2,3 -> 0,1,2 for PyTorch
    y_tr_seq = y_tr_seq - 1
    y_va_seq = y_va_seq - 1
    y_te_seq = y_te_seq - 1

    # Compute class weights to handle class imbalance
    # Higher weight given to minority classes (Up and Down)
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y_tr_seq)
    class_weights_tensor = torch.tensor(
        weights, dtype=torch.float32
    ).to(DEVICE)

    # Convert arrays to PyTorch tensors
    def to_tensors(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    train_ds = TensorDataset(*to_tensors(X_tr_seq, y_tr_seq))
    val_ds   = TensorDataset(*to_tensors(X_va_seq, y_va_seq))
    test_ds  = TensorDataset(*to_tensors(X_te_seq, y_te_seq))

    # Shuffle training data each epoch, keep val/test in order
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"    Sequences — train: {len(train_ds):,}  "
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
    model     : LOB_LSTM
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
    model     : LOB_LSTM
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
    model         : LOB_LSTM
    train_loader  : DataLoader — training data
    val_loader    : DataLoader — validation data
    class_weights : torch.Tensor — per-class loss weights
    max_epochs    : int — maximum number of epochs (50)
    patience      : int — early stopping patience (10)
    lr            : float — Adam learning rate (0.001)

    Returns
    -------
    model   : LOB_LSTM with best weights restored
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
    print("  LSTM — FI-2010 LOB")
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

        # --- Prepare DataLoaders with sliding window sequences ---
        train_loader, val_loader, test_loader, class_weights = \
            prepare_loaders(X_train, y_train, X_val, y_val,
                            X_test,  y_test,  WINDOW, BATCH_SIZE)

        # --- Build model and move to selected device ---
        model = LOB_LSTM(
            input_size  = 144,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            num_classes = 3,
            dropout     = DROPOUT
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"\n    Model parameters: {n_params:,}")

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
        evaluator.record("LSTM", fold, test_true, test_preds)

        # --- Save trained model weights ---
        # To reload later without retraining:
        #   model = LOB_LSTM()
        #   model.load_state_dict(torch.load(f"lstm_fold{fold}.pt"))
        save_path = f"lstm_fold{fold}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"    Model weights saved -> {save_path}")

    # --- Final summary across all folds ---
    evaluator.summary("LSTM")
    evaluator.save("results_lstm.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
