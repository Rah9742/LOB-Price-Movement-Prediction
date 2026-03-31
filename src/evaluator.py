"""
Evaluator — Comprehensive Metrics & Results Saver
=================================================
Shared evaluator for all models.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


CLASS_NAMES = {1: "Up", 2: "Stationary", 3: "Down"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


class Evaluator:
    """Accumulates fold output and saves it under output/<model>/horizon_<k>/."""

    def __init__(self, model_name: str, horizon: int, results_root: str | Path | None = None):
        root = Path(results_root) if results_root else DEFAULT_OUTPUT_DIR
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        self.model_name = model_name
        self.horizon = horizon
        self.results_dir = root / model_name / f"horizon_{horizon}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def record(
        self,
        fold: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: dict | None = None,
        feature_importances: np.ndarray | None = None,
        best_params: dict | None = None,
    ) -> None:
        """Compute metrics for one fold and persist them to disk."""
        labels = [1, 2, 3]
        acc = accuracy_score(y_true, y_pred)
        prec_m = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        rec_m = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1_m = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        prec_c = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rec_c = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        metrics = {
            "fold": fold,
            "accuracy": round(float(acc), 6),
            "macro_precision": round(float(prec_m), 6),
            "macro_recall": round(float(rec_m), 6),
            "macro_f1": round(float(f1_m), 6),
            "per_class": {},
        }
        for i, cls in enumerate(labels):
            metrics["per_class"][CLASS_NAMES[cls]] = {
                "precision": round(float(prec_c[i]), 6),
                "recall": round(float(rec_c[i]), 6),
                "f1": round(float(f1_c[i]), 6),
            }
        if best_params is not None:
            metrics["best_params"] = _jsonify(best_params)

        self.records.append(metrics)
        prefix = self.results_dir / f"fold{fold}"

        with (prefix.parent / f"{prefix.name}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        np.savez_compressed(f"{prefix}_predictions.npz", y_true=y_true, y_pred=y_pred)
        np.save(f"{prefix}_confusion_matrix.npy", cm)

        if history is not None:
            with (prefix.parent / f"{prefix.name}_history.json").open("w", encoding="utf-8") as handle:
                json.dump(_jsonify(history), handle, indent=2)
        if feature_importances is not None:
            np.save(f"{prefix}_feature_importances.npy", feature_importances)
        if best_params is not None:
            with (prefix.parent / f"{prefix.name}_best_params.json").open("w", encoding="utf-8") as handle:
                json.dump(_jsonify(best_params), handle, indent=2)

        print(f"\n    --- Fold {fold} Test Results ---")
        print(f"    Accuracy       : {acc:.4f}")
        print(f"    Macro Precision: {prec_m:.4f}")
        print(f"    Macro Recall   : {rec_m:.4f}")
        print(f"    Macro F1       : {f1_m:.4f}")
        for i, cls in enumerate(labels):
            name = CLASS_NAMES[cls]
            print(f"    {name:>11s}  P={prec_c[i]:.4f}  R={rec_c[i]:.4f}  F1={f1_c[i]:.4f}")
        print(f"    Results saved -> {prefix}_*")

    def summary(self) -> None:
        """Print and save a summary CSV across all recorded folds."""
        if not self.records:
            print("No output recorded yet.")
            return

        print(f"\n{'=' * 60}")
        print(f"  {self.model_name} — Horizon k={self.horizon} — Summary")
        print(f"{'=' * 60}")

        header = [
            "fold", "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "Up_P", "Up_R", "Up_F1",
            "Stat_P", "Stat_R", "Stat_F1",
            "Down_P", "Down_R", "Down_F1",
        ]
        rows = []
        for record in self.records:
            row = [
                record["fold"],
                record["accuracy"],
                record["macro_precision"],
                record["macro_recall"],
                record["macro_f1"],
            ]
            for cls_name in ["Up", "Stationary", "Down"]:
                per_class = record["per_class"][cls_name]
                row.extend([per_class["precision"], per_class["recall"], per_class["f1"]])
            rows.append(row)

        avg = ["AVG"]
        for col_idx in range(1, len(header)):
            avg.append(round(float(np.mean([row[col_idx] for row in rows])), 6))
        rows.append(avg)

        print(f"  {'Fold':>5} {'Acc':>7} {'mPrec':>7} {'mRec':>7} {'mF1':>7}")
        for row in rows:
            print(f"  {str(row[0]):>5} {row[1]:>7.4f} {row[2]:>7.4f} {row[3]:>7.4f} {row[4]:>7.4f}")

        csv_path = self.results_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\n  Summary CSV saved -> {csv_path}")
        print(f"{'=' * 60}\n")


def _jsonify(obj):
    """Convert numpy-heavy objects into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
