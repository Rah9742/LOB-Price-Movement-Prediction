"""
Random Forest — FI-2010 LOB Dataset
===================================
Ensemble of decision trees with validation tuning.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import parse_args
from src.data_loader import load_folds
from src.evaluator import Evaluator


N_ESTIMATORS = [100, 200, 300]
MAX_DEPTH = [5, 10, 20, None]
N_ESTIMATORS_DEBUG = [20]
MAX_DEPTH_DEBUG = [5]
N_JOBS = -1


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_est_grid, depth_grid, verbose=True):
    """Grid-search (n_estimators, max_depth) on validation macro F1."""
    if verbose:
        print(f"    Tuning n_est={n_est_grid}  max_depth={depth_grid} ...")

    best, best_f1 = {"n_estimators": 100, "max_depth": None}, -1.0
    for n_est in n_est_grid:
        for max_d in depth_grid:
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=N_JOBS,
                random_state=42,
            )
            model.fit(X_train, y_train)
            f1 = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
            if verbose:
                depth_str = str(max_d) if max_d is not None else "None"
                print(f"      n_est={n_est:<4} depth={depth_str:<5} val F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best = {"n_estimators": n_est, "max_depth": max_d}

    if verbose:
        print(f"    Best: {best}  val F1={best_f1:.4f}")
    return best


def run(data_dir: str = "./data", horizon: int = 5, debug: bool = False):
    print("=" * 60)
    print(f"  RANDOM FOREST — horizon k={horizon}{'  [DEBUG]' if debug else ''}")
    print("=" * 60 + "\n")

    folds = load_folds(data_dir=data_dir, horizon=horizon, debug=debug)
    evaluator = Evaluator("random_forest", horizon)
    n_grid = N_ESTIMATORS_DEBUG if debug else N_ESTIMATORS
    d_grid = MAX_DEPTH_DEBUG if debug else MAX_DEPTH

    for fd in folds:
        fold = fd["fold"]
        X_train, y_train = fd["train"]
        X_val, y_val = fd["val"]
        X_test, y_test = fd["test"]

        print(f"\n{'=' * 60}")
        print(f"  Fold {fold}")
        print(f"{'=' * 60}")

        best = tune_hyperparameters(X_train, y_train, X_val, y_val, n_grid, d_grid, verbose=True)

        print(f"\n    Retraining on train+val with {best} ...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        model = RandomForestClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=N_JOBS,
            random_state=42,
        )
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)

        evaluator.record(
            fold,
            y_test,
            y_pred,
            feature_importances=model.feature_importances_,
            best_params=best,
        )

    evaluator.summary()
    return evaluator


if __name__ == "__main__":
    args = parse_args("Random Forest — FI-2010 LOB")
    run(args.data_dir, args.horizon, args.debug)
