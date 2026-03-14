"""
Logistic Regression — FI-2010 LOB Dataset
==========================================
A linear classification model that directly optimises for
classification using softmax and cross-entropy loss.

Unlike Ridge Regression which uses +1/-1 encoding and minimises
squared error, Logistic Regression models class probabilities
directly — a more natural fit for classification tasks.

Regularisation parameter C (inverse of λ) is tuned on the
validation set. class_weight='balanced' is used to handle
class imbalance.

Evaluation
----------
- Folds 7, 8, 9 only
- Validation set used to tune C
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds

Usage
-----
python logistic_regression.py

Dependencies: numpy, scikit-learn, data_loader, evaluator
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.data_loader import load_folds
from src.evaluator import Evaluator

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR  = "data"
C_GRID    = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # values to search
MAX_ITER  = 1000   # max iterations for solver convergence
# -------------------------------------------------


# ==================================================
# C TUNING ON VALIDATION SET
# ==================================================

def tune_C(X_train: np.ndarray, y_train: np.ndarray,
           X_val:   np.ndarray, y_val:   np.ndarray,
           C_grid:  list = C_GRID,
           verbose: bool = True) -> float:
    """
    Search over C_grid and return value with best val macro F1.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data
    C_grid           : list of C values to try
    verbose          : print search results

    Returns
    -------
    best_C : float
    """
    if verbose:
        print(f"    Tuning C over {C_grid}...")

    best_C  = C_grid[0]
    best_f1 = -1.0

    for C in C_grid:
        model = LogisticRegression(
            C            = C,
            multi_class  = "multinomial",
            solver       = "lbfgs",
            class_weight = "balanced",
            max_iter     = MAX_ITER,
            random_state = 42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1     = f1_score(y_val, y_pred, average="macro", zero_division=0)

        if verbose:
            print(f"      C={C:.0e}  val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_C  = C

    if verbose:
        print(f"    Best C={best_C:.0e}  val F1={best_f1:.4f}")

    return best_C


# ==================================================
# MAIN TRAINING LOOP
# ==================================================

def run(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  LOGISTIC REGRESSION — FI-2010 LOB")
    print("=" * 60 + "\n")

    folds     = load_folds(data_dir=data_dir)
    evaluator = Evaluator()

    for fold_data in folds:
        fold             = fold_data["fold"]
        X_train, y_train = fold_data["train"]
        X_val,   y_val   = fold_data["val"]
        X_test,  y_test  = fold_data["test"]

        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        # --- Tune C on validation set ---
        best_C = tune_C(X_train, y_train, X_val, y_val,
                        C_grid=C_GRID, verbose=True)

        # --- Retrain on full training data (train + val) using best C ---
        print(f"\n    Retraining on train + val with C={best_C:.0e}...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        model = LogisticRegression(
            C            = best_C,
            multi_class  = "multinomial",
            solver       = "lbfgs",
            class_weight = "balanced",
            max_iter     = MAX_ITER,
            random_state = 42
        )
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)

        # --- Record results ---
        evaluator.record("Logistic Regression", fold, y_test, y_pred)

    # --- Final summary ---
    evaluator.summary("Logistic Regression")
    evaluator.save("results_logistic.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
