"""
Logistic Regression — FI-2010 LOB Dataset
=========================================
Multinomial logistic regression with validation tuning.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import parse_args
from src.data_loader import load_folds
from src.evaluator import Evaluator


C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
C_GRID_DEBUG = [1e-2, 1.0]
MAX_ITER = 1000


def tune_C(X_train, y_train, X_val, y_val, C_grid, verbose=True):
    """Grid-search C on validation macro F1."""
    if verbose:
        print(f"    Tuning C over {C_grid} ...")

    best_C, best_f1 = C_grid[0], -1.0
    for C in C_grid:
        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=MAX_ITER,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if verbose:
            print(f"      C={C:.0e}  val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C = f1, C

    if verbose:
        print(f"    Best C={best_C:.0e}  val F1={best_f1:.4f}")
    return best_C


def run(data_dir: str = "./data", horizon: int = 5, debug: bool = False):
    print("=" * 60)
    print(f"  LOGISTIC REGRESSION — horizon k={horizon}{'  [DEBUG]' if debug else ''}")
    print("=" * 60 + "\n")

    folds = load_folds(data_dir=data_dir, horizon=horizon, debug=debug)
    evaluator = Evaluator("logistic", horizon)
    c_grid = C_GRID_DEBUG if debug else C_GRID

    for fd in folds:
        fold = fd["fold"]
        X_train, y_train = fd["train"]
        X_val, y_val = fd["val"]
        X_test, y_test = fd["test"]

        print(f"\n{'=' * 60}")
        print(f"  Fold {fold}")
        print(f"{'=' * 60}")

        best_C = tune_C(X_train, y_train, X_val, y_val, c_grid, verbose=True)

        print(f"\n    Retraining on train+val with C={best_C:.0e} ...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        model = LogisticRegression(
            C=best_C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=MAX_ITER,
            random_state=42,
        )
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)

        evaluator.record(fold, y_test, y_pred, best_params={"C": best_C})

    evaluator.summary()
    return evaluator


if __name__ == "__main__":
    args = parse_args("Logistic Regression — FI-2010 LOB")
    run(args.data_dir, args.horizon, args.debug)
