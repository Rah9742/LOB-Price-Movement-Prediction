"""
Random Forest — FI-2010 LOB Dataset
=====================================
An ensemble of decision trees where each tree is trained on a
random subset of data and features. Final prediction is by
majority vote across all trees.

Unlike the linear models, Random Forest captures non-linear
relationships and feature interactions without explicit feature
engineering. Also produces feature importances across all 144
features, broken down by feature group (u1-u9).

Hyperparameter tuning on validation set:
    - n_estimators : number of trees
    - max_depth    : maximum depth of each tree

class_weight='balanced' used to handle class imbalance.

Evaluation
----------
- Folds 7, 8, 9 only
- Validation set used to tune hyperparameters
- Test set untouched until final evaluation
- Reports macro F1, accuracy, per-class F1 across all 3 folds
- Feature importance plot saved as PNG

Usage
-----
python random_forest.py

Dependencies: numpy, scikit-learn, matplotlib, data_loader, evaluator
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data_loader import load_folds
from src.evaluator import Evaluator, DEFAULT_REPORTS_DIR

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATA_DIR        = "data"
N_ESTIMATORS    = [100, 200, 300]     # number of trees to search
MAX_DEPTH       = [5, 10, 20, None]   # None = fully grown trees
N_JOBS          = -1                  # use all CPU cores
# -------------------------------------------------

# Feature group boundaries (0-indexed)
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
# HYPERPARAMETER TUNING
# ==================================================

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         X_val:   np.ndarray, y_val:   np.ndarray,
                         n_estimators_grid: list = N_ESTIMATORS,
                         max_depth_grid:    list = MAX_DEPTH,
                         verbose: bool = True) -> dict:
    """
    Grid search over n_estimators and max_depth using val macro F1.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data
    n_estimators_grid : list of n_estimators values
    max_depth_grid    : list of max_depth values
    verbose           : print search results

    Returns
    -------
    best_params : dict with keys 'n_estimators', 'max_depth'
    """
    if verbose:
        print(f"    Tuning n_estimators={n_estimators_grid} "
              f"max_depth={max_depth_grid}...")

    best_params = {"n_estimators": 100, "max_depth": None}
    best_f1     = -1.0

    for n_est in n_estimators_grid:
        for max_d in max_depth_grid:
            model = RandomForestClassifier(
                n_estimators = n_est,
                max_depth    = max_d,
                max_features = "sqrt",
                class_weight = "balanced",
                n_jobs       = N_JOBS,
                random_state = 42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1     = f1_score(y_val, y_pred, average="macro",
                              zero_division=0)

            if verbose:
                depth_str = str(max_d) if max_d else "None"
                print(f"      n_est={n_est:<4}  max_depth={depth_str:<5} "
                      f"val F1={f1:.4f}")

            if f1 > best_f1:
                best_f1     = f1
                best_params = {"n_estimators": n_est, "max_depth": max_d}

    if verbose:
        print(f"    Best params: {best_params}  val F1={best_f1:.4f}")

    return best_params


# ==================================================
# FEATURE IMPORTANCE PLOT
# ==================================================

def plot_feature_importance(importances: np.ndarray,
                            fold: int,
                            out_path: str = None):
    """
    Plot feature importances across all 144 features,
    shaded by feature group (u1-u9).

    Parameters
    ----------
    importances : np.ndarray, shape (144,)
        Feature importances from RandomForestClassifier
    fold : int
        Fold number (for plot title)
    out_path : str
        Path to save PNG (default: plot_rf_importance_fold{fold}.png)
    """
    GROUP_COLORS = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
        "#1abc9c", "#3498db", "#9b59b6", "#e91e63", "#607d8b"
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Random Forest Feature Importances — Fold {fold}\n"
        f"(144 features, shaded by feature group u1–u9)",
        fontsize=12, fontweight="bold"
    )

    # --- Top: per-feature bar chart ---
    ax = axes[0]
    x  = np.arange(144)
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

    # Group labels on x-axis
    for group, (start, end) in FEATURE_GROUPS.items():
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95,
                group.split("(")[0].strip(),
                ha="center", va="top", fontsize=7, rotation=0)

    # --- Bottom: grouped bar chart ---
    ax2    = axes[1]
    groups = list(FEATURE_GROUPS.keys())
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
                 bar.get_height() + 0.001,
                 f"{imp:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = Path(out_path) if out_path else DEFAULT_REPORTS_DIR / f"plot_rf_importance_fold{fold}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Feature importance plot saved -> {out}")
    plt.show()


# ==================================================
# MAIN TRAINING LOOP
# ==================================================

def run(data_dir: str = DATA_DIR):
    print("=" * 60)
    print("  RANDOM FOREST — FI-2010 LOB")
    print("=" * 60 + "\n")

    folds             = load_folds(data_dir=data_dir)
    evaluator         = Evaluator()
    all_importances   = []

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
            n_estimators_grid = N_ESTIMATORS,
            max_depth_grid    = MAX_DEPTH,
            verbose           = True
        )

        # --- Retrain on full training data (train + val) ---
        print(f"\n    Retraining on train + val with {best_params}...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        model = RandomForestClassifier(
            n_estimators = best_params["n_estimators"],
            max_depth    = best_params["max_depth"],
            max_features = "sqrt",
            class_weight = "balanced",
            n_jobs       = N_JOBS,
            random_state = 42
        )
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)

        # --- Record results ---
        evaluator.record("Random Forest", fold, y_test, y_pred)

        # --- Feature importances ---
        importances = model.feature_importances_
        all_importances.append(importances)
        plot_feature_importance(importances, fold)

    # --- Average feature importance across all folds ---
    print("\n  Plotting average feature importance across folds 7, 8, 9...")
    avg_importances = np.mean(all_importances, axis=0)
    plot_feature_importance(avg_importances, fold="avg",
                            out_path=str(DEFAULT_REPORTS_DIR / "plot_rf_importance_avg.png"))

    # --- Final summary ---
    evaluator.summary("Random Forest")
    evaluator.save("results_random_forest.csv")

    return evaluator


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    run(data_dir=data_dir)
