"""
Ridge Regression — FI-2010 LOB Dataset
=======================================
Replicates the paper's baseline Ridge Regression model exactly.

Implementation follows the closed-form solution from the paper:
    W = (X X^T + λI)^-1 X T^T

Where:
    X : feature matrix (144 x n_samples)
    T : label matrix   (3 x n_samples), encoded as +1/-1
    λ : regularisation parameter (tuned on validation set)

Labels are encoded as:
    +1 for the correct class
    -1 for all other classes

Evaluation
----------
- Folds 7, 8, 9 only
- Validation set used to tune λ
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds

Usage
-----
python ridge_regression.py

Dependencies: numpy, scikit-learn, data_loader, evaluator
"""

import numpy as np
from sklearn.metrics import f1_score

from src.data_loader import load_folds
from src.evaluator import Evaluator

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR    = "data"
LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # values to search
# -------------------------------------------------


# ==================================================
# LABEL ENCODING
# ==================================================

def encode_labels(y: np.ndarray, classes: list = [1, 2, 3]) -> np.ndarray:
    """
    Encode integer labels as +1/-1 matrix as per the paper.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Integer labels (1, 2, 3)
    classes : list
        All possible class labels

    Returns
    -------
    T : np.ndarray, shape (n_classes, n_samples)
        +1 where sample belongs to class, -1 otherwise
    """
    T = np.full((len(classes), len(y)), -1, dtype=np.float64)
    for i, cls in enumerate(classes):
        T[i, y == cls] = 1.0
    return T


# ==================================================
# RIDGE REGRESSION — TRAIN & PREDICT
# ==================================================

def train_ridge(X: np.ndarray, T: np.ndarray,
                lam: float) -> np.ndarray:
    """
    Solve for W using the closed-form solution from the paper.

    Uses the form W = (X X^T + λI)^-1 X T^T
    which is efficient when n_features (144) < n_samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Feature matrix — note: transposed vs sklearn convention
    T : np.ndarray, shape (n_classes, n_samples)
        +1/-1 encoded label matrix
    lam : float
        Regularisation parameter λ

    Returns
    -------
    W : np.ndarray, shape (n_features, n_classes)
    """
    n_features = X.shape[0]
    # W = (X X^T + λI)^-1 X T^T
    A = X @ X.T + lam * np.eye(n_features)
    W = np.linalg.solve(A, X @ T.T)
    return W


def predict_ridge(W: np.ndarray, X: np.ndarray,
                  classes: list = [1, 2, 3]) -> np.ndarray:
    """
    Predict class labels using trained weights.

    Prediction rule: argmax of W^T x for each sample.

    Parameters
    ----------
    W : np.ndarray, shape (n_features, n_classes)
    X : np.ndarray, shape (n_features, n_samples)
    classes : list

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples,)
        Predicted class labels (1, 2, 3)
    """
    scores  = W.T @ X                          # shape: (n_classes, n_samples)
    indices = np.argmax(scores, axis=0)        # shape: (n_samples,)
    return np.array(classes)[indices]


# ==================================================
# LAMBDA TUNING ON VALIDATION SET
# ==================================================

def tune_lambda(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray,   y_val: np.ndarray,
                lambda_grid: list = LAMBDA_GRID,
                verbose: bool = True) -> float:
    """
    Search over lambda_grid and return the value with best val F1.

    Parameters
    ----------
    X_train, y_train : training data (sklearn convention: rows=samples)
    X_val,   y_val   : validation data
    lambda_grid      : list of λ values to try
    verbose          : print search results

    Returns
    -------
    best_lam : float
    """
    # Transpose to (n_features, n_samples) for ridge convention
    Xtr = X_train.T
    Xva = X_val.T
    Ttr = encode_labels(y_train)

    if verbose:
        print(f"    Tuning λ over {lambda_grid}...")

    best_lam  = lambda_grid[0]
    best_f1   = -1.0

    for lam in lambda_grid:
        W      = train_ridge(Xtr, Ttr, lam)
        y_pred = predict_ridge(W, Xva)
        f1     = f1_score(y_val, y_pred, average="macro", zero_division=0)

        if verbose:
            print(f"      λ={lam:.0e}  val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1  = f1
            best_lam = lam

    if verbose:
        print(f"    Best λ={best_lam:.0e}  val F1={best_f1:.4f}")

    return best_lam


# ==================================================
# MAIN TRAINING LOOP
# ==================================================

def run(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  RIDGE REGRESSION — FI-2010 LOB")
    print("=" * 60 + "\n")

    # Load data
    folds     = load_folds(data_dir=data_dir)
    evaluator = Evaluator()

    for fold_data in folds:
        fold              = fold_data["fold"]
        X_train, y_train  = fold_data["train"]
        X_val,   y_val    = fold_data["val"]
        X_test,  y_test   = fold_data["test"]

        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        # --- Tune λ on validation set ---
        best_lam = tune_lambda(X_train, y_train, X_val, y_val,
                               lambda_grid=LAMBDA_GRID, verbose=True)

        # --- Retrain on full training data (train + val) using best λ ---
        print(f"\n    Retraining on train + val with λ={best_lam:.0e}...")
        X_full = np.vstack([X_train, X_val])   # shape: (n_tr+n_val, 144)
        y_full = np.concatenate([y_train, y_val])

        W      = train_ridge(X_full.T, encode_labels(y_full), best_lam)
        y_pred = predict_ridge(W, X_test.T)

        # --- Record results ---
        evaluator.record("Ridge Regression", fold, y_test, y_pred)

    # --- Final summary ---
    evaluator.summary("Ridge Regression")
    evaluator.save("results_ridge.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
