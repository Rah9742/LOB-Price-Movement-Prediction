"""Plot saved output from output/<model>/horizon_<k>/."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


ALL_MODELS = ["ridge", "logistic", "random_forest", "xgboost", "mlp", "lstm"]
FOLDS = [7, 8, 9]
CLASS_NAMES = ["Up", "Stationary", "Down"]


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


def plot_confusion_matrices(models, horizon, results_root, out_dir):
    ensure_dir(out_dir)
    for model in models:
        fig, axes = plt.subplots(1, len(FOLDS), figsize=(5 * len(FOLDS), 4.5))
        if len(FOLDS) == 1:
            axes = [axes]
        found = False
        for i, fold in enumerate(FOLDS):
            cm_path = Path(results_root) / model / f"horizon_{horizon}" / f"fold{fold}_confusion_matrix.npy"
            if not cm_path.exists():
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title(f"Fold {fold}")
                continue
            found = True
            cm = np.load(cm_path)
            if HAS_SEABORN:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[i])
            else:
                axes[i].imshow(cm, cmap="Blues")
                for r in range(3):
                    for c in range(3):
                        axes[i].text(c, r, str(cm[r, c]), ha="center", va="center")
                axes[i].set_xticks([0, 1, 2])
                axes[i].set_xticklabels(CLASS_NAMES)
                axes[i].set_yticks([0, 1, 2])
                axes[i].set_yticklabels(CLASS_NAMES)
            axes[i].set_title(f"Fold {fold}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        if found:
            fig.suptitle(f"{model.replace('_', ' ').title()} — Confusion Matrices (k={horizon})", fontsize=13, fontweight="bold")
            plt.tight_layout()
            out = Path(out_dir) / f"cm_{model}_k{horizon}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
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


def plot_comparison(models, horizon, results_root, out_dir):
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
    out = Path(out_dir) / f"comparison_k{horizon}.png"
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
            ])
            labels.append(f"{model}:{fold}")
    if not rows:
        return
    data = np.array(rows)
    plt.figure(figsize=(8, max(4, len(labels) * 0.4)))
    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=CLASS_NAMES, yticklabels=labels)
    else:
        plt.imshow(data, cmap="YlGnBu", aspect="auto")
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        plt.yticks(range(len(labels)), labels)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                plt.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center")
    plt.title(f"Per-Class F1 Heatmap — Horizon k={horizon}")
    plt.tight_layout()
    out = Path(out_dir) / f"heatmap_k{horizon}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot FI-2010 results")
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--plot", choices=["all", "confusion", "loss", "importance", "comparison", "heatmap"], default="all")
    parser.add_argument("--results_root", type=str, default="./output")
    parser.add_argument("--out_dir", type=str, default="./reports")
    args = parser.parse_args()

    if args.plot in ("all", "confusion"):
        plot_confusion_matrices(args.models, args.horizon, args.results_root, args.out_dir)
    if args.plot in ("all", "loss"):
        plot_loss_curves(args.models, args.horizon, args.results_root, args.out_dir)
    if args.plot in ("all", "importance"):
        plot_feature_importance(args.models, args.horizon, args.results_root, args.out_dir)
    if args.plot in ("all", "comparison"):
        plot_comparison(args.models, args.horizon, args.results_root, args.out_dir)
    if args.plot in ("all", "heatmap"):
        plot_heatmap(args.models, args.horizon, args.results_root, args.out_dir)


if __name__ == "__main__":
    main()
