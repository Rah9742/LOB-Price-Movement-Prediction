"""
Ridge Regression — FI-2010 LOB Dataset
======================================
Closed-form ridge classifier with validation tuning.
"""

import numpy as np
from sklearn.metrics import f1_score

from src.config import parse_args
from src.data_loader import load_folds
from src.evaluator import Evaluator


LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LAMBDA_GRID_DEBUG = [1e-3, 1.0]


def encode_labels(y, classes=(1, 2, 3)):
    """Encode labels as a +1/-1 matrix."""
    T = np.full((len(classes), len(y)), -1.0)
    for i, cls in enumerate(classes):
        T[i, y == cls] = 1.0
    return T


def train_ridge(X, T, lam):
    """Solve W = (X X^T + λI)^-1 X T^T."""
    d = X.shape[0]
    A = X @ X.T + lam * np.eye(d)
    return np.linalg.solve(A, X @ T.T)


def predict_ridge(W, X, classes=(1, 2, 3)):
    """Predict labels via argmax(W^T x)."""
    scores = W.T @ X
    return np.array(classes)[np.argmax(scores, axis=0)]


def tune_lambda(X_train, y_train, X_val, y_val, lambda_grid, verbose=True):
    """Grid-search lambda on validation macro F1."""
    Xtr = X_train.T
    Xva = X_val.T
    Ttr = encode_labels(y_train)

    if verbose:
        print(f"    Tuning λ over {lambda_grid} ...")

    best_lam, best_f1 = lambda_grid[0], -1.0
    for lam in lambda_grid:
        W = train_ridge(Xtr, Ttr, lam)
        y_pred = predict_ridge(W, Xva)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if verbose:
            print(f"      λ={lam:.0e}  val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_lam = f1, lam

    if verbose:
        print(f"    Best λ={best_lam:.0e}  val F1={best_f1:.4f}")
    return best_lam


def run(data_dir: str = "./data", horizon: int = 5, debug: bool = False):
    print("=" * 60)
    print(f"  RIDGE REGRESSION — horizon k={horizon}{'  [DEBUG]' if debug else ''}")
    print("=" * 60 + "\n")

    folds = load_folds(data_dir=data_dir, horizon=horizon, debug=debug)
    evaluator = Evaluator("ridge", horizon)
    lam_grid = LAMBDA_GRID_DEBUG if debug else LAMBDA_GRID

    for fd in folds:
        fold = fd["fold"]
        X_train, y_train = fd["train"]
        X_val, y_val = fd["val"]
        X_test, y_test = fd["test"]

        print(f"\n{'=' * 60}")
        print(f"  Fold {fold}")
        print(f"{'=' * 60}")

        best_lam = tune_lambda(X_train, y_train, X_val, y_val, lam_grid, verbose=True)

        print(f"\n    Retraining on train+val with λ={best_lam:.0e} ...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        W = train_ridge(X_full.T, encode_labels(y_full), best_lam)
        y_pred = predict_ridge(W, X_test.T)

        evaluator.record(fold, y_test, y_pred, best_params={"lambda": best_lam})

    evaluator.summary()
    return evaluator


if __name__ == "__main__":
    args = parse_args("Ridge Regression — FI-2010 LOB")
    run(args.data_dir, args.horizon, args.debug)
