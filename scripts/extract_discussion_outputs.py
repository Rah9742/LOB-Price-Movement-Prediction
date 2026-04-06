#!/usr/bin/env python3
"""Extract discussion-ready CSVs from saved experiment artefacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
DISCUSSION_ROOT = PROJECT_ROOT / "reports" / "discussion_outputs"

MODELS = ["ridge", "logistic", "random_forest", "xgboost", "mlp", "lstm"]
HORIZONS = [1, 5, 10]
FOLDS = [7, 8, 9]
CLASS_ORDER = ["Up", "Stationary", "Down"]
DISPLAY = {
    "ridge": "ridge",
    "logistic": "logistic",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "mlp": "mlp",
    "lstm": "lstm",
}
FEATURE_GROUPS = [
    ("u1", "Basic LOB", 0, 40),
    ("u2", "Spread/Mid", 40, 60),
    ("u3", "Price Diff", 60, 82),
    ("u4", "P&V Means", 82, 86),
    ("u5", "Accum Diff", 86, 88),
    ("u6", "P&V Deriv", 88, 128),
    ("u7", "Avg Intens", 128, 134),
    ("u8", "Rel Intens", 134, 140),
    ("u9", "Intens Accel", 140, 144),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_path(model: str, horizon: int, fold: int) -> Path:
    return OUTPUT_ROOT / model / f"horizon_{horizon}" / f"fold{fold}_metrics.json"


def confusion_path(model: str, horizon: int, fold: int) -> Path:
    return OUTPUT_ROOT / model / f"horizon_{horizon}" / f"fold{fold}_confusion_matrix.npy"


def importance_path(model: str, horizon: int, fold: int) -> Path:
    return OUTPUT_ROOT / model / f"horizon_{horizon}" / f"fold{fold}_feature_importances.npy"


def extract_fold_metrics() -> tuple[list[dict], dict, list[str]]:
    rows = []
    metrics_by_key = {}
    missing = []
    for model in MODELS:
        for horizon in HORIZONS:
            for fold in FOLDS:
                path = metric_path(model, horizon, fold)
                if not path.exists():
                    missing.append(str(path.relative_to(PROJECT_ROOT)))
                    continue
                data = load_json(path)
                key = (model, horizon, fold)
                metrics_by_key[key] = data
                rows.append(
                    {
                        "model": DISPLAY[model],
                        "horizon": horizon,
                        "fold": fold,
                        "macro_f1": data["macro_f1"],
                        "accuracy": data["accuracy"],
                    }
                )
    rows.sort(key=lambda r: (r["horizon"], r["model"], r["fold"]))
    return rows, metrics_by_key, missing


def extract_pairwise(metrics_by_key: dict) -> list[dict]:
    comparisons = [
        ("lstm", "xgboost"),
        ("xgboost", "random_forest"),
        ("mlp", "logistic"),
    ]
    rows = []
    for horizon in HORIZONS:
        for left, right in comparisons:
            for fold in FOLDS:
                left_f1 = metrics_by_key[(left, horizon, fold)]["macro_f1"]
                right_f1 = metrics_by_key[(right, horizon, fold)]["macro_f1"]
                diff = round(left_f1 - right_f1, 6)
                if diff > 0:
                    winner = DISPLAY[left]
                elif diff < 0:
                    winner = DISPLAY[right]
                else:
                    winner = "tie"
                rows.append(
                    {
                        "horizon": horizon,
                        "comparison": f"{DISPLAY[left]}_minus_{DISPLAY[right]}",
                        "fold": fold,
                        "left_model": DISPLAY[left],
                        "right_model": DISPLAY[right],
                        "left_macro_f1": left_f1,
                        "right_macro_f1": right_f1,
                        "macro_f1_difference": diff,
                        "winner": winner,
                    }
                )
    return rows


def extract_horizon_summary(metrics_by_key: dict) -> list[dict]:
    rows = []
    for model in MODELS:
        means = {}
        for horizon in HORIZONS:
            vals = [metrics_by_key[(model, horizon, fold)]["macro_f1"] for fold in FOLDS]
            means[horizon] = round(float(np.mean(vals)), 6)
        best_horizon = max(HORIZONS, key=lambda h: means[h])
        rows.append(
            {
                "model": DISPLAY[model],
                "mean_macro_f1_k1": means[1],
                "mean_macro_f1_k5": means[5],
                "mean_macro_f1_k10": means[10],
                "delta_k5_minus_k1": round(means[5] - means[1], 6),
                "delta_k10_minus_k5": round(means[10] - means[5], 6),
                "best_horizon_by_mean_macro_f1": best_horizon,
            }
        )
    rows.sort(key=lambda r: r["model"])
    return rows


def extract_lstm_classwise(metrics_by_key: dict) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for horizon in HORIZONS:
        for fold in FOLDS:
            metric_file = metric_path("lstm", horizon, fold)
            cm_file = confusion_path("lstm", horizon, fold)
            if not metric_file.exists() or not cm_file.exists():
                missing.extend(
                    str(p.relative_to(PROJECT_ROOT))
                    for p in [metric_file, cm_file]
                    if not p.exists()
                )
                continue
            metrics = metrics_by_key[("lstm", horizon, fold)]
            cm = np.load(cm_file)
            for class_name in CLASS_ORDER:
                row = {
                    "horizon": horizon,
                    "fold": fold,
                    "class": class_name,
                    "precision": metrics["per_class"][class_name]["precision"],
                    "recall": metrics["per_class"][class_name]["recall"],
                    "f1": metrics["per_class"][class_name]["f1"],
                    "cm_true_up_pred_up": int(cm[0, 0]),
                    "cm_true_up_pred_stationary": int(cm[0, 1]),
                    "cm_true_up_pred_down": int(cm[0, 2]),
                    "cm_true_stationary_pred_up": int(cm[1, 0]),
                    "cm_true_stationary_pred_stationary": int(cm[1, 1]),
                    "cm_true_stationary_pred_down": int(cm[1, 2]),
                    "cm_true_down_pred_up": int(cm[2, 0]),
                    "cm_true_down_pred_stationary": int(cm[2, 1]),
                    "cm_true_down_pred_down": int(cm[2, 2]),
                }
                rows.append(row)
    rows.sort(key=lambda r: (r["horizon"], r["fold"], CLASS_ORDER.index(r["class"])))
    return rows, missing


def mean_importances(model: str, horizon: int) -> tuple[np.ndarray | None, list[str]]:
    arrays = []
    missing = []
    for fold in FOLDS:
        path = importance_path(model, horizon, fold)
        if path.exists():
            arrays.append(np.load(path))
        else:
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    if not arrays:
        return None, missing
    return np.mean(np.vstack(arrays), axis=0), missing


def extract_group_importance(model: str) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for horizon in HORIZONS:
        avg_imp, horizon_missing = mean_importances(model, horizon)
        missing.extend(horizon_missing)
        if avg_imp is None:
            continue
        group_rows = []
        for group_id, group_name, start, end in FEATURE_GROUPS:
            total = round(float(avg_imp[start:end].sum()), 6)
            group_rows.append(
                {
                    "horizon": horizon,
                    "item_type": "group",
                    "item_id": group_id,
                    "item_name": group_name,
                    "rank": None,
                    "importance": total,
                    "start_feature_index_1based": start + 1,
                    "end_feature_index_1based": end,
                }
            )
        group_rows.sort(key=lambda r: r["importance"], reverse=True)
        for rank, row in enumerate(group_rows, start=1):
            row["rank"] = rank
            rows.append(row)

        top_idx = np.argsort(avg_imp)[::-1][:10]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "horizon": horizon,
                    "item_type": "feature",
                    "item_id": f"f{idx + 1}",
                    "item_name": f"feature_{idx + 1}",
                    "rank": rank,
                    "importance": round(float(avg_imp[idx]), 6),
                    "start_feature_index_1based": int(idx + 1),
                    "end_feature_index_1based": int(idx + 1),
                }
            )
    rows.sort(key=lambda r: (r["horizon"], r["item_type"], r["rank"]))
    return rows, missing


def rank_lookup(rows: list[dict], horizon: int, item_type: str = "group") -> list[dict]:
    return [r for r in rows if r["horizon"] == horizon and r["item_type"] == item_type]


def build_text_summary(metrics_by_key: dict, xgb_rows: list[dict], rf_rows: list[dict]) -> str:
    lines = []

    lines.append("Available artefacts inspected:")
    lines.append("- Per-fold metrics JSONs for all models, horizons 1/5/10, folds 7/8/9")
    lines.append("- Per-fold predictions NPZs for all models")
    lines.append("- Per-fold confusion matrices for all models")
    lines.append("- Per-fold feature-importance arrays for XGBoost and Random Forest")
    lines.append("- Summary CSVs and report plots under reports/")
    lines.append("- No separate feature-name registry found; top individual features are reported as feature indices")
    lines.append("")

    lines.append("Next-best model after LSTM by horizon (mean macro F1 across folds 7-9):")
    for horizon in HORIZONS:
        scored = []
        for model in MODELS:
            vals = [metrics_by_key[(model, horizon, fold)]["macro_f1"] for fold in FOLDS]
            scored.append((float(np.mean(vals)), DISPLAY[model]))
        scored.sort(reverse=True)
        lines.append(f"- k={horizon}: {scored[1][1]}")
    lines.append("")

    lines.append("Did LSTM beat XGBoost on all three folds?")
    for horizon in HORIZONS:
        wins = []
        for fold in FOLDS:
            left = metrics_by_key[("lstm", horizon, fold)]["macro_f1"]
            right = metrics_by_key[("xgboost", horizon, fold)]["macro_f1"]
            wins.append(left > right)
        lines.append(f"- k={horizon}: {'yes' if all(wins) else 'no'}")
    lines.append("")

    lines.append("Did XGBoost beat Random Forest on all three folds?")
    for horizon in HORIZONS:
        wins = []
        for fold in FOLDS:
            left = metrics_by_key[("xgboost", horizon, fold)]["macro_f1"]
            right = metrics_by_key[("random_forest", horizon, fold)]["macro_f1"]
            wins.append(left > right)
        lines.append(f"- k={horizon}: {'yes' if all(wins) else 'no'}")
    lines.append("")

    lines.append("Best horizon by mean macro F1:")
    for model in MODELS:
        means = {h: float(np.mean([metrics_by_key[(model, h, fold)]["macro_f1"] for fold in FOLDS])) for h in HORIZONS}
        best_horizon = max(HORIZONS, key=lambda h: means[h])
        lines.append(f"- {DISPLAY[model]}: k={best_horizon}")
    lines.append("")

    lines.append("LSTM error pattern check:")
    for horizon in HORIZONS:
        agg = np.zeros((3, 3), dtype=int)
        for fold in FOLDS:
            agg += np.load(confusion_path('lstm', horizon, fold))
        movement_vs_stationary = int(agg[0, 1] + agg[1, 0] + agg[2, 1] + agg[1, 2])
        up_vs_down = int(agg[0, 2] + agg[2, 0])
        if movement_vs_stationary > up_vs_down:
            label = "movement vs stationary errors were more common"
        elif movement_vs_stationary < up_vs_down:
            label = "up vs down errors were more common"
        else:
            label = "movement vs stationary and up vs down errors were equal"
        lines.append(
            f"- k={horizon}: {label} "
            f"(movement-vs-stationary={movement_vs_stationary}, up-vs-down={up_vs_down})"
        )
    lines.append("")

    lines.append("XGBoost feature-importance stability:")
    xgb_top = {}
    for horizon in HORIZONS:
        ranked = rank_lookup(xgb_rows, horizon, "group")
        xgb_top[horizon] = [r["item_id"] for r in ranked]
        lines.append(f"- k={horizon}: {' > '.join(xgb_top[horizon])}")
    stable = xgb_top[1] == xgb_top[5] == xgb_top[10]
    u6_first_u1_second = all(order[0] == "u6" and order[1] == "u1" for order in xgb_top.values())
    lines.append(f"- Ranking identical across horizons: {'yes' if stable else 'no'}")
    lines.append(f"- u6 ranked first and u1 ranked second at all horizons: {'yes' if u6_first_u1_second else 'no'}")
    lines.append("")

    if rf_rows:
        lines.append("Random Forest grouped ranking compared with XGBoost:")
        for horizon in HORIZONS:
            ranked = rank_lookup(rf_rows, horizon, "group")
            rf_order = [r["item_id"] for r in ranked]
            similarity = "similar" if rf_order[:3] == xgb_top[horizon][:3] else "not identical"
            lines.append(f"- k={horizon}: {' > '.join(rf_order)} ({similarity} top-3 pattern)")

    return "\n".join(lines)


def main() -> None:
    fold_rows, metrics_by_key, missing_metrics = extract_fold_metrics()
    pairwise_rows = extract_pairwise(metrics_by_key)
    horizon_rows = extract_horizon_summary(metrics_by_key)
    lstm_rows, missing_lstm = extract_lstm_classwise(metrics_by_key)
    xgb_rows, missing_xgb = extract_group_importance("xgboost")
    rf_rows, missing_rf = extract_group_importance("random_forest")

    write_csv(
        DISCUSSION_ROOT / "fold_level_metrics.csv",
        fold_rows,
        ["model", "horizon", "fold", "macro_f1", "accuracy"],
    )
    write_csv(
        DISCUSSION_ROOT / "pairwise_fold_level_comparisons.csv",
        pairwise_rows,
        [
            "horizon",
            "comparison",
            "fold",
            "left_model",
            "right_model",
            "left_macro_f1",
            "right_macro_f1",
            "macro_f1_difference",
            "winner",
        ],
    )
    write_csv(
        DISCUSSION_ROOT / "horizon_summary.csv",
        horizon_rows,
        [
            "model",
            "mean_macro_f1_k1",
            "mean_macro_f1_k5",
            "mean_macro_f1_k10",
            "delta_k5_minus_k1",
            "delta_k10_minus_k5",
            "best_horizon_by_mean_macro_f1",
        ],
    )
    if lstm_rows:
        write_csv(
            DISCUSSION_ROOT / "lstm_classwise_metrics.csv",
            lstm_rows,
            [
                "horizon",
                "fold",
                "class",
                "precision",
                "recall",
                "f1",
                "cm_true_up_pred_up",
                "cm_true_up_pred_stationary",
                "cm_true_up_pred_down",
                "cm_true_stationary_pred_up",
                "cm_true_stationary_pred_stationary",
                "cm_true_stationary_pred_down",
                "cm_true_down_pred_up",
                "cm_true_down_pred_stationary",
                "cm_true_down_pred_down",
            ],
        )
    if xgb_rows:
        write_csv(
            DISCUSSION_ROOT / "xgboost_feature_importance_summary.csv",
            xgb_rows,
            [
                "horizon",
                "item_type",
                "item_id",
                "item_name",
                "rank",
                "importance",
                "start_feature_index_1based",
                "end_feature_index_1based",
            ],
        )
    if rf_rows:
        write_csv(
            DISCUSSION_ROOT / "random_forest_grouped_feature_importance.csv",
            [r for r in rf_rows if r["item_type"] == "group"],
            [
                "horizon",
                "item_type",
                "item_id",
                "item_name",
                "rank",
                "importance",
                "start_feature_index_1based",
                "end_feature_index_1based",
            ],
        )

    summary_text = build_text_summary(metrics_by_key, xgb_rows, [r for r in rf_rows if r["item_type"] == "group"])
    summary_path = DISCUSSION_ROOT / "discussion_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    missing = {
        "missing_metrics": missing_metrics,
        "missing_lstm": missing_lstm,
        "missing_xgboost_importances": missing_xgb,
        "missing_random_forest_importances": missing_rf,
    }
    missing_path = DISCUSSION_ROOT / "missing_artefacts.json"
    missing_path.write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print(summary_text)
    print("")
    print(f"Saved CSVs under {DISCUSSION_ROOT}")
    print(f"Missing artefact log: {missing_path}")


if __name__ == "__main__":
    main()
