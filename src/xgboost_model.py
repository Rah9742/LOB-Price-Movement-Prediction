"""
XGBoost — FI-2010 LOB Dataset
=============================
Gradient-boosted trees with validation tuning and saved feature importances.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import parse_args
from src.data_loader import load_folds
from src.evaluator import Evaluator


N_ESTIMATORS = [100, 200, 300]
MAX_DEPTH = [3, 6, 10]
LEARNING_RATE = [0.01, 0.05, 0.1]
EARLY_STOPPING = 20
N_ESTIMATORS_DEBUG = [30]
MAX_DEPTH_DEBUG = [3]
LEARNING_RATE_DEBUG = [0.1]


def to_xgb(y):
    return y - 1


def from_xgb(y):
    return y + 1


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_est_grid, depth_grid, lr_grid, verbose=True):
    """Grid-search XGBoost hyperparameters on validation macro F1."""
    if verbose:
        total = len(n_est_grid) * len(depth_grid) * len(lr_grid)
        print(f"    Tuning over {total} combinations ...")

    y_tr = to_xgb(y_train)
    y_va = to_xgb(y_val)
    sample_weights = compute_sample_weight("balanced", y_tr)
    best = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    best_f1 = -1.0

    for n_est in n_est_grid:
        for max_d in depth_grid:
            for lr in lr_grid:
                model = XGBClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    objective="multi:softmax",
                    num_class=3,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    early_stopping_rounds=EARLY_STOPPING,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_train, y_tr, sample_weight=sample_weights, eval_set=[(X_val, y_va)], verbose=False)
                y_pred = from_xgb(model.predict(X_val))
                f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
                if verbose:
                    print(f"      n_est={n_est:<4} depth={max_d:<3} lr={lr:<5} val F1={f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    best = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}

    if verbose:
        print(f"    Best: {best}  val F1={best_f1:.4f}")
    return best


def run(data_dir: str = "./data", horizon: int = 5, debug: bool = False):
    print("=" * 60)
    print(f"  XGBOOST — horizon k={horizon}{'  [DEBUG]' if debug else ''}")
    print("=" * 60 + "\n")

    folds = load_folds(data_dir=data_dir, horizon=horizon, debug=debug)
    evaluator = Evaluator("xgboost", horizon)
    n_grid = N_ESTIMATORS_DEBUG if debug else N_ESTIMATORS
    d_grid = MAX_DEPTH_DEBUG if debug else MAX_DEPTH
    lr_grid = LEARNING_RATE_DEBUG if debug else LEARNING_RATE

    for fd in folds:
        fold = fd["fold"]
        X_train, y_train = fd["train"]
        X_val, y_val = fd["val"]
        X_test, y_test = fd["test"]

        print(f"\n{'=' * 60}")
        print(f"  Fold {fold}")
        print(f"{'=' * 60}")

        best = tune_hyperparameters(X_train, y_train, X_val, y_val, n_grid, d_grid, lr_grid, verbose=True)

        print(f"\n    Retraining on train+val with {best} ...")
        X_full = np.vstack([X_train, X_val])
        y_full = to_xgb(np.concatenate([y_train, y_val]))
        sample_weights = compute_sample_weight("balanced", y_full)
        model = XGBClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            objective="multi:softmax",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_full, y_full, sample_weight=sample_weights, verbose=False)
        y_pred = from_xgb(model.predict(X_test))

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
    args = parse_args("XGBoost — FI-2010 LOB")
    run(args.data_dir, args.horizon, args.debug)
