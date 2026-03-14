"""
Evaluation Loop — FI-2010 LOB Dataset
======================================
Shared evaluation module used by all five models:
  - Ridge Regression
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LSTM

Each model plugs into the same evaluation loop, ensuring
fair and consistent comparison across all models.

Metrics reported per fold
-------------------------
- Macro F1 score (primary metric)
- Accuracy
- Per-class F1 for Up (1), Stationary (2), Down (3)
- Confusion matrix

Metrics reported overall
------------------------
- Mean ± std across folds 7, 8, 9 for all metrics
- Final summary comparison table across all models

Usage
-----
from src.evaluator import Evaluator

evaluator = Evaluator()

# Option A: record live predictions during training
evaluator.record(
    model_name = "Ridge Regression",
    fold       = 9,
    y_true     = y_test,
    y_pred     = predictions
)

# Option B: load results from saved CSV files
evaluator.load_all_results({
    "Ridge Regression"   : "results_ridge.csv",
    "Logistic Regression": "results_logistic.csv",
    "Random Forest"      : "results_random_forest.csv",
    "XGBoost"            : "results_xgboost.csv",
    "LSTM"               : "results_lstm.csv",
})

# Print summary for one model
evaluator.summary("Ridge Regression")

# Print comparison table across all models
evaluator.compare_all()

# Save results to CSV
evaluator.save("results.csv")

Dependencies: numpy, scikit-learn
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# EVALUATOR CLASS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


class Evaluator:
    """
    Collects and reports evaluation metrics across folds and models.
    """

    CLASS_NAMES = {1: "Up", 2: "Stationary", 3: "Down"}

    def __init__(self, reports_dir: str | Path | None = None):
        # Storage: {model_name: [fold_result_dict, ...]}
        self._results = {}
        reports_path = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
        self.reports_dir = (reports_path if reports_path.is_absolute()
                            else PROJECT_ROOT / reports_path)

    def _resolve_report_path(self, filepath: str | Path, create_parent: bool = False) -> Path:
        """Resolve result files, defaulting bare filenames into reports/."""
        path = Path(filepath)
        if not path.is_absolute() and len(path.parts) == 1:
            path = self.reports_dir / path
        elif not path.is_absolute():
            path = PROJECT_ROOT / path
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------------------------------
    # RECORDING RESULTS
    # --------------------------------------------------

    def record(self,
               model_name: str,
               fold: int,
               y_true: np.ndarray,
               y_pred: np.ndarray):
        """
        Record predictions for one model on one fold.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g. 'Ridge Regression')
        fold : int
            Fold number (7, 8, or 9)
        y_true : np.ndarray
            True labels (integers 1, 2, 3)
        y_pred : np.ndarray
            Predicted labels (integers 1, 2, 3)
        """
        if model_name not in self._results:
            self._results[model_name] = []

        labels       = [1, 2, 3]
        f1_per_class = f1_score(
            y_true, y_pred, labels=labels,
            average=None, zero_division=0
        )

        fold_result = {
            "fold"        : fold,
            "accuracy"    : accuracy_score(y_true, y_pred),
            "f1_macro"    : f1_score(y_true, y_pred, average="macro",
                                     zero_division=0),
            "f1_up"       : f1_per_class[0],
            "f1_stat"     : f1_per_class[1],
            "f1_down"     : f1_per_class[2],
            "precision"   : precision_score(y_true, y_pred, average="macro",
                                            zero_division=0),
            "recall"      : recall_score(y_true, y_pred, average="macro",
                                         zero_division=0),
            "conf_matrix" : confusion_matrix(y_true, y_pred, labels=labels),
            "y_true"      : y_true,
            "y_pred"      : y_pred,
        }

        self._results[model_name].append(fold_result)
        self._print_fold_result(model_name, fold_result)

    # --------------------------------------------------
    # LOADING RESULTS FROM CSV
    # --------------------------------------------------

    def load_results(self, filepath: str):
        """
        Load results from a previously saved CSV file into
        this Evaluator instance.

        Allows results from multiple model scripts to be
        combined into one Evaluator for compare_all().

        Parameters
        ----------
        filepath : str
            Path to a results CSV file (e.g. 'results_ridge.csv')
        """
        path = self._resolve_report_path(filepath)
        if not path.exists():
            print(f"  WARNING: Could not find '{filepath}' — skipping.")
            return

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model"]
                if model_name not in self._results:
                    self._results[model_name] = []

                self._results[model_name].append({
                    "fold"        : int(row["fold"]),
                    "accuracy"    : float(row["accuracy"]),
                    "f1_macro"    : float(row["f1_macro"]),
                    "f1_up"       : float(row["f1_up"]),
                    "f1_stat"     : float(row["f1_stat"]),
                    "f1_down"     : float(row["f1_down"]),
                    "precision"   : float(row["precision"]),
                    "recall"      : float(row["recall"]),
                    "conf_matrix" : np.zeros((3, 3), dtype=int),
                    "y_true"      : np.array([]),
                    "y_pred"      : np.array([]),
                })

        print(f"  Loaded '{path}'")

    def load_all_results(self, model_files: dict):
        """
        Load results from multiple CSV files into this Evaluator.

        Parameters
        ----------
        model_files : dict
            Maps model name -> CSV filepath. Example:
            {
                "Ridge Regression"   : "results_ridge.csv",
                "Logistic Regression": "results_logistic.csv",
                "Random Forest"      : "results_random_forest.csv",
                "XGBoost"            : "results_xgboost.csv",
                "LSTM"               : "results_lstm.csv",
            }
        """
        print("Loading results from CSVs...")
        for filepath in model_files.values():
            self.load_results(filepath)
        print()

    # --------------------------------------------------
    # PER-FOLD PRINTING
    # --------------------------------------------------

    def _print_fold_result(self, model_name: str, result: dict):
        fold = result["fold"]
        print(f"\n  [{model_name}] Fold {fold} results:")
        print(f"    Accuracy   : {result['accuracy']:.4f}")
        print(f"    F1 Macro   : {result['f1_macro']:.4f}")
        print(f"    F1 Up      : {result['f1_up']:.4f}")
        print(f"    F1 Stat    : {result['f1_stat']:.4f}")
        print(f"    F1 Down    : {result['f1_down']:.4f}")
        print(f"    Precision  : {result['precision']:.4f}")
        print(f"    Recall     : {result['recall']:.4f}")
        if result["conf_matrix"].sum() > 0:
            print(f"\n    Confusion Matrix (rows=true, cols=pred):")
            print(f"    {'':12} {'Up':>8} {'Stat':>8} {'Down':>8}")
            cm = result["conf_matrix"]
            for i, cls in enumerate(["Up", "Stat", "Down"]):
                print(f"    {cls:<12} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

    # --------------------------------------------------
    # PER-MODEL SUMMARY
    # --------------------------------------------------

    def summary(self, model_name: str):
        """
        Print mean ± std across all recorded folds for one model.

        Parameters
        ----------
        model_name : str
        """
        if model_name not in self._results:
            print(f"No results recorded for '{model_name}'")
            return

        results = self._results[model_name]
        folds   = [r["fold"] for r in results]

        print("\n" + "=" * 60)
        print(f"  SUMMARY — {model_name}")
        print(f"  Folds: {folds}")
        print("=" * 60)

        metrics = ["accuracy", "f1_macro", "f1_up", "f1_stat",
                   "f1_down", "precision", "recall"]
        labels  = ["Accuracy  ", "F1 Macro  ", "F1 Up     ",
                   "F1 Stat   ", "F1 Down   ", "Precision ",
                   "Recall    "]

        for metric, label in zip(metrics, labels):
            vals = np.array([r[metric] for r in results])
            print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f}  "
                  f"(folds: {[f'{v:.4f}' for v in vals]})")

        # Aggregated confusion matrix across all folds
        agg_cm = sum(r["conf_matrix"] for r in results)
        if agg_cm.sum() > 0:
            print(f"\n  Aggregated Confusion Matrix (rows=true, cols=pred):")
            print(f"  {'':12} {'Up':>8} {'Stat':>8} {'Down':>8}")
            for i, cls in enumerate(["Up", "Stat", "Down"]):
                print(f"  {cls:<12} {agg_cm[i,0]:>8} {agg_cm[i,1]:>8} "
                      f"{agg_cm[i,2]:>8}")

        print("=" * 60 + "\n")

    # --------------------------------------------------
    # CROSS-MODEL COMPARISON TABLE
    # --------------------------------------------------

    def compare_all(self):
        """
        Print a comparison table of mean ± std across all
        registered models. Works whether results were recorded
        live or loaded from CSV files.
        """
        if not self._results:
            print("No results recorded yet.")
            return

        print("\n" + "=" * 90)
        print("  MODEL COMPARISON — Mean ± Std across folds")
        print("=" * 90)
        print(f"  {'Model':<22} {'Accuracy':>14} {'F1 Macro':>14} "
              f"{'F1 Up':>10} {'F1 Stat':>10} {'F1 Down':>10}")
        print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*10} {'-'*10} {'-'*10}")

        for model_name, results in self._results.items():
            acc   = np.array([r["accuracy"] for r in results])
            f1    = np.array([r["f1_macro"] for r in results])
            f1_up = np.array([r["f1_up"]    for r in results])
            f1_st = np.array([r["f1_stat"]  for r in results])
            f1_dn = np.array([r["f1_down"]  for r in results])

            print(
                f"  {model_name:<22} "
                f"{acc.mean():.3f}±{acc.std():.3f}  "
                f"{f1.mean():.3f}±{f1.std():.3f}  "
                f"{f1_up.mean():.3f}±{f1_up.std():.3f}  "
                f"{f1_st.mean():.3f}±{f1_st.std():.3f}  "
                f"{f1_dn.mean():.3f}±{f1_dn.std():.3f}"
            )

        print("=" * 90 + "\n")

    # --------------------------------------------------
    # SAVE TO CSV
    # --------------------------------------------------

    def save(self, filepath: str = "results.csv"):
        """
        Save all recorded results to a CSV file.

        Parameters
        ----------
        filepath : str
            Output file path (default: results.csv)
        """
        rows = []
        for model_name, results in self._results.items():
            for r in results:
                rows.append({
                    "model"     : model_name,
                    "fold"      : r["fold"],
                    "accuracy"  : round(r["accuracy"],  4),
                    "f1_macro"  : round(r["f1_macro"],  4),
                    "f1_up"     : round(r["f1_up"],     4),
                    "f1_stat"   : round(r["f1_stat"],   4),
                    "f1_down"   : round(r["f1_down"],   4),
                    "precision" : round(r["precision"], 4),
                    "recall"    : round(r["recall"],    4),
                })

        if not rows:
            print("No results to save.")
            return

        fieldnames = ["model", "fold", "accuracy", "f1_macro",
                      "f1_up", "f1_stat", "f1_down", "precision", "recall"]

        path = self._resolve_report_path(filepath, create_parent=True)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results saved to '{path}'")

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def get_results(self, model_name: str) -> list:
        """Return raw results list for a model."""
        return self._results.get(model_name, [])

    def models(self) -> list:
        """Return list of all registered model names."""
        return list(self._results.keys())


# ==================================================
# STANDALONE METRIC HELPERS
# (can be used outside the Evaluator class)
# ==================================================

def evaluate_predictions(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          verbose: bool = True) -> dict:
    """
    Compute all metrics for a single set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (1, 2, 3)
    y_pred : np.ndarray
        Predicted labels (1, 2, 3)
    verbose : bool
        Print sklearn classification report if True

    Returns
    -------
    dict of metric name -> value
    """
    labels       = [1, 2, 3]
    f1_per_class = f1_score(y_true, y_pred, labels=labels,
                            average=None, zero_division=0)

    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1_macro"  : f1_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "f1_up"     : f1_per_class[0],
        "f1_stat"   : f1_per_class[1],
        "f1_down"   : f1_per_class[2],
        "precision" : precision_score(y_true, y_pred, average="macro",
                                      zero_division=0),
        "recall"    : recall_score(y_true, y_pred, average="macro",
                                   zero_division=0),
    }

    if verbose:
        print(classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=["Up", "Stationary", "Down"],
            zero_division=0
        ))

    return metrics


# ==================================================
# RUN DIRECTLY — loads all model CSVs and compares
# ==================================================

if __name__ == "__main__":
    evaluator = Evaluator()

    evaluator.load_all_results({
        "Ridge Regression"   : "results_ridge.csv",
        "Logistic Regression": "results_logistic.csv",
        "MLP"                : "results_mlp.csv",
        "Random Forest"      : "results_random_forest.csv",
        "XGBoost"            : "results_xgboost.csv",
        "LSTM"               : "results_lstm.csv",
    })

    for model_name in evaluator.models():
        evaluator.summary(model_name)

    evaluator.compare_all()
    evaluator.save("results_all_models.csv")
