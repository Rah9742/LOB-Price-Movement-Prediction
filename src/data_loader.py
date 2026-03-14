"""
Data Loading Module — FI-2010 LOB Dataset
==========================================
Shared module used by all five models:
  - Ridge Regression
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LSTM

Setup
-----
- Folds 7, 8, 9 only
- Label: row 148 (0-indexed: row 147) = horizon k=5
- Features: all 144 rows (0-indexed: rows 0-143)
- Validation: final training day when it can be inferred from the
  preceding test fold, otherwise fallback to the historical 1/fold split
- Test set: untouched until final evaluation

Directory structure expected
----------------------------
your_project/
├── data/
│   ├── Train_Dst_NoAuction_ZScore_CF_1.txt
│   ├── Test_Dst_NoAuction_ZScore_CF_1.txt
│   ...
│   ├── Train_Dst_NoAuction_ZScore_CF_9.txt
│   └── Test_Dst_NoAuction_ZScore_CF_9.txt
└── data_loader.py

Usage
-----
from src.data_loader import load_folds, get_fold_data

folds = load_folds(data_dir="./data")

for fold_data in folds:
    X_train, y_train = fold_data["train"]
    X_val,   y_val   = fold_data["val"]
    X_test,  y_test  = fold_data["test"]
    fold_num         = fold_data["fold"]

Dependencies: numpy
"""

import numpy as np
from pathlib import Path

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
FOLDS_TO_USE  = [7, 8, 9]       # only use later folds
LABEL_ROW     = 147             # 0-indexed row 147 = paper row 148 = horizon k=5
FEATURE_ROWS  = slice(0, 144)   # rows 0-143 = all 144 features
N_FEATURES    = 144
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
# -------------------------------------------------


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve relative data paths from the project root."""
    path = Path(data_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_raw_fold(data_dir: str, split: str, fold: int) -> np.ndarray:
    """
    Load a single raw .txt file.

    Parameters
    ----------
    data_dir : str
        Path to folder containing the .txt files.
    split : str
        'Train' or 'Test'
    fold : int
        Fold number (1-9)

    Returns
    -------
    np.ndarray, shape (149, n_samples)
        Full data matrix — rows 0-143 are features, rows 144-148 are labels.
    """
    fname = f"{split}_Dst_NoAuction_ZScore_CF_{fold}.txt"
    path  = resolve_data_dir(data_dir) / fname
    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find: {path}\n"
            f"Check data_dir='{data_dir}' and that the file follows the\n"
            f"naming convention: Train/Test_Dst_NoAuction_ZScore_CF_N.txt"
        )

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the dataset contents. "
            "Fetch the real data with `git lfs pull` before training."
        )

    return np.loadtxt(path)


def extract_features_labels(data: np.ndarray, label_row: int = LABEL_ROW):
    """
    Split a raw data matrix into features X and labels y.

    Parameters
    ----------
    data : np.ndarray, shape (149, n_samples)
    label_row : int
        0-indexed row to use as label (default 147 = horizon k=5)

    Returns
    -------
    X : np.ndarray, shape (n_samples, 144)
        Feature matrix — transposed so rows = samples, cols = features.
    y : np.ndarray, shape (n_samples,)
        Integer labels: 1=Up, 2=Stationary, 3=Down
    """
    X = data[FEATURE_ROWS, :].T          # shape: (n_samples, 144)
    y = data[label_row, :].astype(int)   # shape: (n_samples,)
    return X, y


def train_val_split(X: np.ndarray, y: np.ndarray, fold: int):
    """
    Carve the last 1/fold fraction of training data as validation.

    This is a fallback used when the exact size of the last training
    day cannot be inferred from the preceding fold.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 144)
    y : np.ndarray, shape (n_samples,)
    fold : int
        Used to compute val_fraction = 1/fold

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    n          = X.shape[0]
    val_frac   = 1.0 / fold
    split_idx  = int(n * (1.0 - val_frac))

    X_train = X[:split_idx, :]
    y_train = y[:split_idx]
    X_val   = X[split_idx:, :]
    y_val   = y[split_idx:]

    return X_train, y_train, X_val, y_val


def infer_validation_size(data_dir: str,
                          fold: int,
                          known_test_sizes: dict[int, int] | None = None) -> int | None:
    """
    Infer the size of the last training day for fold k.

    For FI-2010, `Train_CF_k` contains days 1..k and `Test_CF_{k-1}`
    contains day k. When the previous test file is available, its size
    gives the exact size of the last day that should be used as validation.

    Fold 7 has no preceding test file in this repo, so callers may need
    to fall back to the historical 1/k proportional split.
    """
    previous_fold = fold - 1
    if known_test_sizes and previous_fold in known_test_sizes:
        return known_test_sizes[previous_fold]

    previous_test = resolve_data_dir(data_dir) / (
        f"Test_Dst_NoAuction_ZScore_CF_{previous_fold}.txt"
    )
    if not previous_test.exists():
        return None

    raw_previous_test = np.loadtxt(previous_test)
    return raw_previous_test.shape[1]


def load_folds(data_dir: str = "./data",
               folds_to_use: list = FOLDS_TO_USE,
               label_row: int = LABEL_ROW,
               verbose: bool = True) -> list:
    """
    Load all required folds and split into train / val / test.

    Parameters
    ----------
    data_dir : str
        Path to folder containing the .txt files.
    folds_to_use : list
        Which folds to load (default [7, 8, 9]).
    label_row : int
        0-indexed label row (default 147 = horizon k=5).
    verbose : bool
        Print loading summary if True.

    Returns
    -------
    list of dict, one per fold:
        {
            "fold"  : int,
            "train" : (X_train, y_train),   # shape: (n_tr, 144), (n_tr,)
            "val"   : (X_val,   y_val),     # shape: (n_val, 144), (n_val,)
            "test"  : (X_test,  y_test),    # shape: (n_te, 144), (n_te,)
        }
    """
    if verbose:
        print(f"Loading folds {folds_to_use} from '{data_dir}'")
        print(f"Label row: {label_row} (0-indexed) = horizon k=5")
        print("-" * 60)

    folds = []
    test_sizes = {}
    for fold in folds_to_use:
        # Load raw files
        raw_train = load_raw_fold(data_dir, "Train", fold)
        raw_test  = load_raw_fold(data_dir, "Test",  fold)

        # Extract features and labels
        X_all, y_all   = extract_features_labels(raw_train, label_row)
        X_test, y_test = extract_features_labels(raw_test,  label_row)
        test_sizes[fold] = X_test.shape[0]

        # Split training into train + val
        validation_size = infer_validation_size(
            data_dir, fold, known_test_sizes=test_sizes
        )
        if validation_size is None:
            X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, fold)
            split_note = f"fallback 1/{fold} split (previous fold unavailable)"
        else:
            split_idx = X_all.shape[0] - validation_size
            if split_idx <= 0:
                raise ValueError(
                    f"Invalid validation size {validation_size} for fold {fold}"
                )
            X_train = X_all[:split_idx, :]
            y_train = y_all[:split_idx]
            X_val   = X_all[split_idx:, :]
            y_val   = y_all[split_idx:]
            split_note = (
                f"last day inferred from Test_Dst_NoAuction_ZScore_CF_{fold - 1}.txt"
            )

        if verbose:
            n_tr  = X_train.shape[0]
            n_val = X_val.shape[0]
            n_te  = X_test.shape[0]
            print(f"  Fold {fold}:")
            print(f"    Train : {n_tr:>7,} samples")
            print(f"    Val   : {n_val:>7,} samples  ({split_note})")
            print(f"    Test  : {n_te:>7,} samples")
            _print_class_dist("    Train label dist", y_train)
            _print_class_dist("    Val   label dist", y_val)
            _print_class_dist("    Test  label dist", y_test)
            print()

        folds.append({
            "fold"  : fold,
            "train" : (X_train, y_train),
            "val"   : (X_val,   y_val),
            "test"  : (X_test,  y_test),
        })

    if verbose:
        print("-" * 60)
        print("Data loading complete.\n")

    return folds


def get_fold_data(folds: list, fold_num: int) -> dict:
    """
    Retrieve a specific fold by number.

    Parameters
    ----------
    folds : list
        Output of load_folds()
    fold_num : int
        Fold number to retrieve (e.g. 7, 8, or 9)

    Returns
    -------
    dict with keys: fold, train, val, test
    """
    match = [f for f in folds if f["fold"] == fold_num]
    if not match:
        raise ValueError(
            f"Fold {fold_num} not found. Available: "
            f"{[f['fold'] for f in folds]}"
        )
    return match[0]


def make_sequences(X: np.ndarray, y: np.ndarray,
                   window: int = 100) -> tuple:
    """
    Create sliding window sequences for the LSTM.

    Each sample becomes a (window, 144) sequence.
    The label is taken from the last timestep of each window.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 144)
    y : np.ndarray, shape (n_samples,)
    window : int
        Number of timesteps per sequence (default 100, same as DeepLOB)

    Returns
    -------
    X_seq : np.ndarray, shape (n_samples - window + 1, window, 144)
    y_seq : np.ndarray, shape (n_samples - window + 1,)
    """
    n        = X.shape[0]
    if n < window:
        raise ValueError(
            f"Cannot create window={window} sequences from only {n} samples"
        )
    n_seq    = n - window + 1
    X_seq    = np.zeros((n_seq, window, X.shape[1]), dtype=np.float32)

    for i in range(n_seq):
        X_seq[i] = X[i : i + window, :]

    y_seq = y[window - 1:]   # label aligns to last timestep of window

    return X_seq, y_seq


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights inversely proportional to class frequency.
    Useful for handling class imbalance in sklearn models and PyTorch.

    Parameters
    ----------
    y : np.ndarray
        Integer labels (1, 2, 3)

    Returns
    -------
    dict mapping class label -> weight
    """
    classes, counts = np.unique(y, return_counts=True)
    total           = len(y)
    weights         = {int(c): total / (len(classes) * cnt)
                       for c, cnt in zip(classes, counts)}
    return weights


# ==================================================
# INTERNAL HELPERS
# ==================================================

def _print_class_dist(label: str, y: np.ndarray):
    total = len(y)
    up    = 100 * np.sum(y == 1) / total
    st    = 100 * np.sum(y == 2) / total
    dn    = 100 * np.sum(y == 3) / total
    print(f"{label}: Up={up:.1f}%  Stat={st:.1f}%  Down={dn:.1f}%")


# ==================================================
# QUICK TEST — run this file directly to verify
# ==================================================

if __name__ == "__main__":
    import sys

    data_dir = "data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=" * 60)
    print("  DATA LOADER — QUICK VERIFICATION")
    print("=" * 60 + "\n")

    folds = load_folds(data_dir=data_dir)

    # Verify shapes
    print("Shape verification:")
    for fd in folds:
        X_tr, y_tr = fd["train"]
        X_val, y_val = fd["val"]
        X_te, y_te = fd["test"]
        print(f"  Fold {fd['fold']}:")
        print(f"    X_train: {X_tr.shape}   y_train: {y_tr.shape}")
        print(f"    X_val:   {X_val.shape}   y_val:   {y_val.shape}")
        print(f"    X_test:  {X_te.shape}   y_test:  {y_te.shape}")

    # Verify LSTM sequences
    print("\nLSTM sequence verification (fold 9):")
    fd        = get_fold_data(folds, 9)
    X_tr, y_tr = fd["train"]
    X_seq, y_seq = make_sequences(X_tr, y_tr, window=100)
    print(f"  X_seq shape: {X_seq.shape}")
    print(f"  y_seq shape: {y_seq.shape}")

    # Verify class weights
    print("\nClass weights (fold 9 train):")
    weights = compute_class_weights(y_tr)
    for cls, w in weights.items():
        print(f"  Class {cls}: {w:.4f}")

    print("\nAll checks passed.")
