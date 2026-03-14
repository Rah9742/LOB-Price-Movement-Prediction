"""
XGBoost — FI-2010 LOB Dataset
==============================
A gradient boosted tree model that builds trees sequentially,
where each new tree focuses on correcting the errors made by
the previous ones.

Generally the strongest traditional ML model on structured
tabular data. Key advantages over Random Forest:
- Handles class imbalance via sample weights
- More accurate with proper tuning
- Built-in early stopping

Hyperparameter tuning on validation set (matching Random Forest):
    - n_estimators : number of boosting rounds
    - max_depth    : maximum depth of each tree
    - learning_rate: step size shrinkage

Final training reuses the tuned hyperparameters on train+val
without early stopping so the tuning and final fit use the
same training regime.

Evaluation
----------
- Folds 7, 8, 9 only
- Validation set used to tune hyperparameters
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds
- Feature importance plot saved as PNG

Usage
-----
python xgboost_model.py

Dependencies: numpy, xgboost, scikit-learn, matplotlib,
              data_loader, evaluator
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_folds
from src.evaluator import Evaluator, DEFAULT_REPORTS_DIR

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR        = "data"
N_ESTIMATORS    = [100, 200, 300]       # number of boosting rounds
MAX_DEPTH       = [3, 6, 10]            # tree depth
LEARNING_RATE   = [0.01, 0.05, 0.1]    # step size
# -------------------------------------------------

# Feature group boundaries (0-indexed) — same as Random Forest
FEATURE_GROUPS = {
    "u1 Basic LOB (raw)"          : (0,   40),
    "u2 Spread & Mid-price"        : (40,  60),
    "u3 Price Differences"         : (60,  82),
    "u4 Price & Vol Means"         : (82,  86),
    "u5 Accumulated Differences"   : (86,  88),
    "u6 Price & Vol Derivatives"   : (88,  128),
    "u7 Avg Intensity"             : (128, 134),
    "u8 Relative Intensity"        : (134, 140),
    "u9 Intensity Acceleration"    : (140, 144),
}


# ==================================================
# LABEL REMAPPING
# XGBoost requires 0-indexed labels (0, 1, 2)
# ==================================================

def to_xgb_labels(y: np.ndarray) -> np.ndarray:
    """Remap labels 1,2,3 -> 0,1,2 for XGBoost."""
    return y - 1

def from_xgb_labels(y: np.ndarray) -> np.ndarray:
    """Remap labels 0,1,2 -> 1,2,3 back to original."""
    return y + 1


# ==================================================
# HYPERPARAMETER TUNING
# ==================================================

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         X_val:   np.ndarray, y_val:   np.ndarray,
                         n_estimators_grid: list = N_ESTIMATORS,
                         max_depth_grid:    list = MAX_DEPTH,
                         learning_rate_grid: list = LEARNING_RATE,
                         verbose: bool = True) -> dict:
    """
    Grid search over n_estimators, max_depth, learning_rate
    using validation macro F1.

    Parameters
    ----------
    X_train, y_train : training data (labels 1,2,3)
    X_val,   y_val   : validation data (labels 1,2,3)
    verbose          : print search results

    Returns
    -------
    best_params : dict
    """
    if verbose:
        total = (len(n_estimators_grid) * len(max_depth_grid) *
                 len(learning_rate_grid))
        print(f"    Tuning over {total} combinations...")

    # Remap labels to 0-indexed
    y_tr  = to_xgb_labels(y_train)
    # Sample weights for class imbalance
    sample_weights = compute_sample_weight("balanced", y_tr)

    best_params = {
        "n_estimators" : 100,
        "max_depth"    : 6,
        "learning_rate": 0.1
    }
    best_f1 = -1.0

    for n_est in n_estimators_grid:
        for max_d in max_depth_grid:
            for lr in learning_rate_grid:
                model = XGBClassifier(
                    n_estimators          = n_est,
                    max_depth             = max_d,
                    learning_rate         = lr,
                    objective             = "multi:softmax",
                    num_class             = 3,
                    use_label_encoder     = False,
                    eval_metric           = "mlogloss",
                    n_jobs                = -1,
                    random_state          = 42,
                    verbosity             = 0
                )
                model.fit(
                    X_train, y_tr,
                    sample_weight = sample_weights,
                    verbose       = False
                )
                y_pred = from_xgb_labels(model.predict(X_val))
                f1     = f1_score(y_val, y_pred, average="macro",
                                  zero_division=0)

                if verbose:
                    print(f"      n_est={n_est:<4}  depth={max_d:<3} "
                          f"lr={lr:<5}  val F1={f1:.4f}")

                if f1 > best_f1:
                    best_f1     = f1
                    best_params = {
                        "n_estimators" : n_est,
                        "max_depth"    : max_d,
                        "learning_rate": lr
                    }

    if verbose:
        print(f"    Best params: {best_params}  val F1={best_f1:.4f}")

    return best_params


# ==================================================
# FEATURE IMPORTANCE PLOT
# ==================================================

def plot_feature_importance(importances: np.ndarray,
                            fold,
                            out_path: str = None):
    """
    Plot XGBoost feature importances across all 144 features,
    shaded by feature group (u1-u9).

    Parameters
    ----------
    importances : np.ndarray, shape (144,)
    fold        : int or str (fold number or 'avg')
    out_path    : str, path to save PNG
    """
    GROUP_COLORS = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
        "#1abc9c", "#3498db", "#9b59b6", "#e91e63", "#607d8b"
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"XGBoost Feature Importances — Fold {fold}\n"
        f"(144 features, shaded by feature group u1–u9)",
        fontsize=12, fontweight="bold"
    )

    # --- Top: per-feature bar chart ---
    ax     = axes[0]
    x      = np.arange(144)
    colors = np.empty(144, dtype=object)

    for i, (group, (start, end)) in enumerate(FEATURE_GROUPS.items()):
        colors[start:end] = GROUP_COLORS[i]
        ax.axvspan(start, end, alpha=0.08, color=GROUP_COLORS[i])

    ax.bar(x, importances, color=colors, alpha=0.85, width=1.0)
    ax.set_ylabel("Importance", fontsize=10)
    ax.set_xlim(-1, 144)
    ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for group, (start, end) in FEATURE_GROUPS.items():
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95,
                group.split("(")[0].strip(),
                ha="center", va="top", fontsize=7)

    # --- Bottom: grouped totals ---
    ax2       = axes[1]
    groups    = list(FEATURE_GROUPS.keys())
    group_imp = [importances[s:e].sum()
                 for s, e in FEATURE_GROUPS.values()]
    short_labels = [g.split(" ")[0] for g in groups]

    bars = ax2.bar(short_labels, group_imp,
                   color=GROUP_COLORS[:len(groups)], alpha=0.85)
    ax2.set_ylabel("Total Importance", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    for bar, imp in zip(bars, group_imp):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0005,
                 f"{imp:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = Path(out_path) if out_path else DEFAULT_REPORTS_DIR / f"plot_xgb_importance_fold{fold}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Feature importance plot saved -> {out}")
    plt.show()


# ==================================================
# MAIN TRAINING LOOP
# ==================================================

def run(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  XGBOOST — FI-2010 LOB")
    print("=" * 60 + "\n")

    folds           = load_folds(data_dir=data_dir)
    evaluator       = Evaluator()
    all_importances = []

    for fold_data in folds:
        fold             = fold_data["fold"]
        X_train, y_train = fold_data["train"]
        X_val,   y_val   = fold_data["val"]
        X_test,  y_test  = fold_data["test"]

        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        # --- Tune hyperparameters on validation set ---
        best_params = tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            n_estimators_grid  = N_ESTIMATORS,
            max_depth_grid     = MAX_DEPTH,
            learning_rate_grid = LEARNING_RATE,
            verbose            = True
        )

        # --- Retrain on full training data (train + val) ---
        print(f"\n    Retraining on train + val with {best_params}...")
        X_full = np.vstack([X_train, X_val])
        y_full = to_xgb_labels(np.concatenate([y_train, y_val]))
        sample_weights = compute_sample_weight("balanced", y_full)

        model = XGBClassifier(
            n_estimators      = best_params["n_estimators"],
            max_depth         = best_params["max_depth"],
            learning_rate     = best_params["learning_rate"],
            objective         = "multi:softmax",
            num_class         = 3,
            use_label_encoder = False,
            eval_metric       = "mlogloss",
            n_jobs            = -1,
            random_state      = 42,
            verbosity         = 0
        )
        model.fit(
            X_full, y_full,
            sample_weight = sample_weights,
            verbose       = False
        )

        y_pred = from_xgb_labels(model.predict(X_test))

        # --- Record results ---
        evaluator.record("XGBoost", fold, y_test, y_pred)

        # --- Feature importances ---
        importances = model.feature_importances_
        all_importances.append(importances)
        plot_feature_importance(importances, fold)

    # --- Average feature importance across all folds ---
    print("\n  Plotting average feature importance across folds 7, 8, 9...")
    avg_importances = np.mean(all_importances, axis=0)
    plot_feature_importance(avg_importances, fold="avg",
                            out_path=str(DEFAULT_REPORTS_DIR / "plot_xgb_importance_avg.png"))

    # --- Final summary ---
    evaluator.summary("XGBoost")
    evaluator.save("results_xgboost.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
