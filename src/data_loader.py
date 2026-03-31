"""
Data Loading Module — FI-2010 LOB Dataset
=========================================
Shared module used by all model scripts.

Supports:
- folds 7, 8, 9
- multi-horizon labels k in {1, 2, 3, 5, 10}
- exact validation split via Train_CF_(k-1) when available
- debug truncation for fast end-to-end runs
"""

from pathlib import Path

import numpy as np


HORIZON_TO_ROW = {
    1: 144,
    2: 145,
    3: 146,
    5: 147,
    10: 148,
}

FOLDS_TO_USE = [7, 8, 9]
FEATURE_ROWS = slice(0, 144)
N_FEATURES = 144
DEBUG_SAMPLES = 5000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve relative data paths from the project root."""
    path = Path(data_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_raw_file(data_dir: str | Path, prefix: str, fold: int) -> np.ndarray:
    """Load a FI-2010 train/test text file."""
    path = resolve_data_dir(data_dir) / f"{prefix}_Dst_NoAuction_ZScore_CF_{fold}.txt"
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


def extract_features_labels(data: np.ndarray, label_row: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a raw FI-2010 matrix into X and y."""
    X = data[FEATURE_ROWS, :].T
    y = data[label_row, :].astype(int)
    return X, y


def _truncate(X: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the first n samples for debug mode."""
    return X[:n], y[:n]


def _count_samples(data_dir: str | Path, prefix: str, fold: int) -> int:
    """
    Count samples in a text file by reading the first line.

    Returns -1 when the file does not exist.
    """
    path = resolve_data_dir(data_dir) / f"{prefix}_Dst_NoAuction_ZScore_CF_{fold}.txt"
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    return len(first_line.split())


def train_val_split(X: np.ndarray, y: np.ndarray, fold: int) -> tuple[np.ndarray, ...]:
    """Fallback validation split using the final 1/fold fraction."""
    split_idx = int(X.shape[0] * (1.0 - 1.0 / fold))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def load_folds(
    data_dir: str = "./data",
    horizon: int = 5,
    folds_to_use: list[int] | None = None,
    debug: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Load folds and split each train set into train/validation.

    Validation uses the exact final day boundary when Train_CF_(fold-1) exists.
    Otherwise it falls back to the historical 1/fold split.
    """
    if folds_to_use is None:
        folds_to_use = FOLDS_TO_USE
    if horizon not in HORIZON_TO_ROW:
        raise ValueError(
            f"horizon must be one of {list(HORIZON_TO_ROW.keys())}, got {horizon}"
        )

    label_row = HORIZON_TO_ROW[horizon]

    if verbose:
        mode = "DEBUG" if debug else "FULL"
        print(
            f"Loading folds {folds_to_use} from '{data_dir}' | "
            f"horizon k={horizon} (label row {label_row}) | mode={mode}"
        )
        print("-" * 60)

    folds = []
    for fold in folds_to_use:
        raw_train = load_raw_file(data_dir, "Train", fold)
        raw_test = load_raw_file(data_dir, "Test", fold)

        X_all, y_all = extract_features_labels(raw_train, label_row)
        X_test, y_test = extract_features_labels(raw_test, label_row)

        prev_count = _count_samples(data_dir, "Train", fold - 1)
        if 0 < prev_count < X_all.shape[0]:
            split_idx = prev_count
            split_note = f"exact split via Train_CF_{fold - 1} ({prev_count:,} samples)"
            X_train = X_all[:split_idx]
            y_train = y_all[:split_idx]
            X_val = X_all[split_idx:]
            y_val = y_all[split_idx:]
        else:
            X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, fold)
            split_note = f"fallback 1/{fold} split"

        if debug:
            X_train, y_train = _truncate(X_train, y_train, DEBUG_SAMPLES)
            X_val, y_val = _truncate(X_val, y_val, DEBUG_SAMPLES)
            X_test, y_test = _truncate(X_test, y_test, DEBUG_SAMPLES)

        if verbose:
            print(f"  Fold {fold}:")
            print(f"    Train : {X_train.shape[0]:>7,} samples")
            print(f"    Val   : {X_val.shape[0]:>7,} samples  ({split_note})")
            print(f"    Test  : {X_test.shape[0]:>7,} samples")
            _print_class_dist("    Train labels", y_train)
            _print_class_dist("    Val   labels", y_val)
            _print_class_dist("    Test  labels", y_test)
            print()

        folds.append(
            {
                "fold": fold,
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
            }
        )

    if verbose:
        print("Finished loading all folds.\n")
    return folds


def get_fold_data(folds: list[dict], fold_num: int) -> dict:
    """Retrieve one fold dict by fold number."""
    match = [f for f in folds if f["fold"] == fold_num]
    if not match:
        raise ValueError(
            f"Fold {fold_num} not found. Available: {[f['fold'] for f in folds]}"
        )
    return match[0]


def make_sequences(X: np.ndarray, y: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for recurrent models."""
    n = X.shape[0]
    if n < window:
        raise ValueError(f"Cannot create window={window} sequences from only {n} samples")

    n_seq = n - window + 1
    X_seq = np.zeros((n_seq, window, X.shape[1]), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X[i:i + window, :]
    y_seq = y[window - 1:]
    return X_seq, y_seq


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for labels 1, 2, 3."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}


def _print_class_dist(label: str, y: np.ndarray) -> None:
    total = len(y)
    up = 100 * np.sum(y == 1) / total
    st = 100 * np.sum(y == 2) / total
    dn = 100 * np.sum(y == 3) / total
    print(f"{label}: Up={up:.1f}%  Stat={st:.1f}%  Down={dn:.1f}%")


if __name__ == "__main__":
    data_dir = "./data"
    print("=" * 60)
    print("  DATA LOADER — VERIFICATION")
    print("=" * 60 + "\n")

    for horizon in [1, 5, 10]:
        print(f">>> Horizon k={horizon}")
        folds = load_folds(data_dir=data_dir, horizon=horizon, debug=False)
        for fd in folds:
            Xtr, _ = fd["train"]
            Xva, _ = fd["val"]
            Xte, _ = fd["test"]
            print(
                f"  Fold {fd['fold']}: "
                f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}"
            )
