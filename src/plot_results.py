"""Plot saved output from output/<model>/horizon_<k>/ for one or more horizons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
    }
)

ALL_MODELS = ["ridge", "logistic", "random_forest", "xgboost", "mlp", "LSTM"]
FOLDS = [7, 8, 9]
CLASS_NAMES = ["Up", "Stationary", "Down"]
MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "logistic": "Logistic",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "LSTM": "LSTM",
}
PLOT_CHOICES = [
    "all",
    "confusion",
    "loss",
    "importance",
    "comparison",
    "comparison_avg",
    "comparison_folds",
    "heatmap",
]


FEATURE_GROUPS = {
    "u1 Basic LOB": (0, 40),
    "u2 Spread/Mid": (40, 60),
    "u3 Price Diff": (60, 82),
    "u4 P&V Means": (82, 86),
    "u5 Accum Diff": (86, 88),
    "u6 P&V Deriv": (88, 128),
    "u7 Avg Intens": (128, 134),
    "u8 Rel Intens": (134, 140),
    "u9 Intens Accel": (140, 144),
}
GROUP_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6", "#e91e63", "#607d8b"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _annotate_matrix(ax, data, fmt, threshold=None):
    """Add contrast-aware text annotations to a heatmap."""
    if threshold is None:
        threshold = (float(np.max(data)) + float(np.min(data))) / 2.0
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            color = "white" if value >= threshold else "#1f1f1f"
            ax.text(col, row, format(value, fmt), ha="center", va="center", color=color, fontsize=16)


def _draw_heatmap(ax, data, xlabels, ylabels, title, cmap, fmt, colorbar=True, aspect="auto", vmin=None, vmax=None):
    """Render a heatmap with adaptive text labels and optional colorbar."""
    image = ax.imshow(data, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    _annotate_matrix(ax, data, fmt)
    if colorbar:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image


def _load_summary_rows(path):
    """Load summary rows keyed by fold from one model summary CSV."""
    rows = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fold = row["fold"]
            rows[fold] = row
    return rows


def plot_confusion_matrices(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    for model in models:
        matrices = []
        for fold in FOLDS:
            cm_path = Path(results_root) / model / f"horizon_{horizon}" / f"fold{fold}_confusion_matrix.npy"
            if not cm_path.exists():
                matrices.append(None)
                continue
            cm = np.load(cm_path).astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            normalized_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
            matrices.append(normalized_cm)

        if not any(cm is not None for cm in matrices):
            continue

        valid_matrices = [cm for cm in matrices if cm is not None]
        vmax = 1.0
        vmin = 0.0

        fig, axes = plt.subplots(
            1,
            len(FOLDS),
            figsize=(10.8, 3.55),
            gridspec_kw={"wspace": 0.04},
            constrained_layout=True,
        )
        if len(FOLDS) == 1:
            axes = [axes]

        last_image = None
        for i, (fold, cm) in enumerate(zip(FOLDS, matrices)):
            if cm is None:
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title(f"Fold {fold}")
                axes[i].set_axis_off()
                continue

            last_image = _draw_heatmap(
                axes[i],
                cm,
                CLASS_NAMES,
                CLASS_NAMES,
                f"Fold {fold}",
                "PuBuGn",
                ".3f",
                colorbar=False,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            axes[i].tick_params(axis="y", labelrotation=90, pad=2)
            for label in axes[i].get_yticklabels():
                label.set_verticalalignment("center")
                label.set_horizontalalignment("center")
                x_pos, y_pos = label.get_position()
                label.set_position((x_pos - 0.03, y_pos))
            if i == 0:
                axes[i].set_ylabel("True", fontsize=11, labelpad=16)
            else:
                axes[i].set_ylabel("")
            axes[i].set_xlabel("")
        fig.supxlabel("Predicted", fontsize=11)
        fig.suptitle(
            f"{MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title())} — Confusion Matrices (Horizon {horizon})",
            fontsize=15,
            fontweight="bold",
        )
        if last_image is not None:
            cbar = fig.colorbar(last_image, ax=axes, location="right", fraction=0.022, pad=0.02)
            cbar.ax.set_ylabel("Proportion", rotation=90, labelpad=8)
            out = Path(out_dir) / f"cm_{model}_k{horizon}.png"
            plt.savefig(out, dpi=720, bbox_inches="tight")
            print(f"  Saved: {out}")
        plt.close(fig)


def plot_loss_curves(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    for model in [m for m in models if m in ("mlp", "lstm")]:
        fig, axes = plt.subplots(len(FOLDS), 2, figsize=(12, 4 * len(FOLDS)))
        if len(FOLDS) == 1:
            axes = axes.reshape(1, -1)
        found = False
        for i, fold in enumerate(FOLDS):
            hist_path = Path(results_root) / model / f"horizon_{horizon}" / f"fold{fold}_history.json"
            if not hist_path.exists():
                continue
            found = True
            with hist_path.open("r", encoding="utf-8") as handle:
                hist = json.load(handle)
            epochs = range(1, len(hist["train_loss"]) + 1)
            axes[i, 0].plot(epochs, hist["train_loss"], label="Train")
            axes[i, 0].plot(epochs, hist["val_loss"], label="Val")
            axes[i, 0].set_title(f"Fold {fold} — Loss")
            axes[i, 0].set_xlabel("Epoch")
            axes[i, 0].set_ylabel("Loss")
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(epochs, hist["val_f1"], color="green")
            axes[i, 1].set_title(f"Fold {fold} — Val Macro F1")
            axes[i, 1].set_xlabel("Epoch")
            axes[i, 1].set_ylabel("Macro F1")
            axes[i, 1].grid(True, alpha=0.3)
        if found:
            fig.suptitle(f"{model.upper()} — Training Curves (k={horizon})", fontsize=13, fontweight="bold")
            plt.tight_layout()
            out = Path(out_dir) / f"loss_{model}_k{horizon}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
        plt.close(fig)


def plot_feature_importance(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    for model in [m for m in models if m in ("random_forest", "xgboost")]:
        importances = []
        for fold in FOLDS:
            path = Path(results_root) / model / f"horizon_{horizon}" / f"fold{fold}_feature_importances.npy"
            if path.exists():
                importances.append(np.load(path))
        if not importances:
            continue
        avg_imp = np.mean(importances, axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"{model.replace('_', ' ').title()} — Feature Importances (k={horizon})", fontsize=12, fontweight="bold")
        ax = axes[0]
        x = np.arange(144)
        colors = np.empty(144, dtype=object)
        for i, (group, (start, end)) in enumerate(FEATURE_GROUPS.items()):
            colors[start:end] = GROUP_COLORS[i]
            ax.axvspan(start, end, alpha=0.08, color=GROUP_COLORS[i])
            ax.text((start + end) / 2, max(avg_imp) * 0.95 if np.max(avg_imp) > 0 else 0.01, group, ha="center", va="top", fontsize=7)
        ax.bar(x, avg_imp, color=colors, alpha=0.85, width=1.0)
        ax.set_ylabel("Importance")
        ax.set_xlim(-1, 144)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        ax2 = axes[1]
        short_labels = [g.split(" ")[0] for g in FEATURE_GROUPS]
        group_imp = [avg_imp[s:e].sum() for s, e in FEATURE_GROUPS.values()]
        bars = ax2.bar(short_labels, group_imp, color=GROUP_COLORS[:len(group_imp)], alpha=0.85)
        ax2.set_ylabel("Total Importance")
        ax2.yaxis.grid(True, alpha=0.3)
        for bar, imp in zip(bars, group_imp):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{imp:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        out = Path(out_dir) / f"importance_{model}_k{horizon}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close(fig)


def plot_comparison_avg(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    labels = []
    scores = []
    for model in models:
        path = Path(results_root) / model / f"horizon_{horizon}" / "summary.csv"
        if not path.exists():
            continue
        data = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        avg_row = data[-1]
        labels.append(model.replace("_", " ").title())
        scores.append(float(avg_row[4]))
    if not scores:
        return
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, scores, color="#3498db")
    plt.ylabel("Macro F1")
    plt.title(f"Model Comparison — Horizon k={horizon}")
    plt.ylim(0, max(scores) * 1.1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out = Path(out_dir) / f"comparison_avg_k{horizon}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_comparison_folds(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    labels = [f"Fold {fold}" for fold in FOLDS]
    scores_by_model = []
    model_labels = []
    for model in models:
        path = Path(results_root) / model / f"horizon_{horizon}" / "summary.csv"
        if not path.exists():
            continue
        rows = _load_summary_rows(path)
        fold_scores = []
        complete = True
        for fold in FOLDS:
            key = str(fold)
            if key not in rows:
                complete = False
                break
            fold_scores.append(float(rows[key]["macro_f1"]))
        if complete:
            scores_by_model.append(fold_scores)
            model_labels.append(model.replace("_", " ").title())
    if not scores_by_model:
        return

    x = np.arange(len(FOLDS))
    width = 0.8 / len(scores_by_model)
    plt.figure(figsize=(12, 6))
    for idx, (label, scores) in enumerate(zip(model_labels, scores_by_model)):
        offset = (idx - (len(scores_by_model) - 1) / 2) * width
        bars = plt.bar(x + offset, scores, width=width, label=label)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, labels)
    plt.xlabel("Fold")
    plt.ylabel("Macro F1")
    plt.title(f"Model Comparison — Macro F1 (k={horizon})")
    plt.ylim(0, max(max(scores) for scores in scores_by_model) * 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = Path(out_dir) / f"comparison_folds_k{horizon}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_heatmap(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    rows = []
    labels = []
    for model in models:
        for fold in FOLDS:
            path = Path(results_root) / model / f"horizon_{horizon}" / f"fold{fold}_metrics.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            rows.append([
                metrics["per_class"]["Up"]["f1"],
                metrics["per_class"]["Stationary"]["f1"],
                metrics["per_class"]["Down"]["f1"],
                metrics["macro_f1"],
            ])
            labels.append(f"{model}:{fold}")
    if not rows:
        return
    data = np.array(rows)
    plt.figure(figsize=(9, max(4, len(labels) * 0.4)))
    ax = plt.gca()
    _draw_heatmap(
        ax,
        data,
        ["Up", "Stationary", "Down", "Macro F1"],
        labels,
        f"Per-Class F1 Heatmap — Horizon k={horizon}",
        "YlGnBu",
        ".3f",
        colorbar=True,
        aspect="auto",
    )
    plt.tight_layout()
    out = Path(out_dir) / f"heatmap_k{horizon}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def generate_plots(models, horizons, results_root, out_dir, plot_type):
    """Generate the requested plot set for each horizon."""
    for horizon in horizons:
        print(f"\nGenerating {plot_type} plots for horizon k={horizon}...")
        if plot_type in ("all", "confusion"):
            plot_confusion_matrices(models, horizon, results_root, out_dir)
        if plot_type in ("all", "loss"):
            plot_loss_curves(models, horizon, results_root, out_dir)
        if plot_type in ("all", "importance"):
            plot_feature_importance(models, horizon, results_root, out_dir)
        if plot_type in ("all", "comparison", "comparison_avg"):
            plot_comparison_avg(models, horizon, results_root, out_dir)
        if plot_type in ("all", "comparison", "comparison_folds"):
            plot_comparison_folds(models, horizon, results_root, out_dir)
        if plot_type in ("all", "heatmap"):
            plot_heatmap(models, horizon, results_root, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot FI-2010 results")
    parser.add_argument("--horizon", type=int, nargs="+", default=[5], choices=[1, 5, 10])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--plot", choices=PLOT_CHOICES, default="all")
    parser.add_argument("--results_root", type=str, default="./output")
    parser.add_argument("--out_dir", type=str, default="./reports")
    args = parser.parse_args()
    generate_plots(args.models, args.horizon, args.results_root, args.out_dir, args.plot)


if __name__ == "__main__":
    main()
