"""Cross-horizon comparison plots for table-style model summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("./output")
REPORTS_DIR = Path("./reports")
HORIZONS = [1, 5, 10]
MODELS = [
    ("ridge", "Ridge Regression"),
    ("logistic", "Logistic Regression"),
    ("mlp", "MLP"),
    ("random_forest", "Random Forest"),
    ("xgboost", "XGBoost"),
    ("lstm", "LSTM"),
]
METRICS = [
    ("macro_f1", "F1 Macro"),
    ("accuracy", "Accuracy"),
    ("macro_precision", "Precision"),
    ("macro_recall", "Recall"),
]
MODEL_COLORS = {
    "Ridge Regression": "#4c78a8",
    "Logistic Regression": "#f58518",
    "MLP": "#54a24b",
    "Random Forest": "#e45756",
    "XGBoost": "#b279a2",
    "LSTM": "#48A999",
}
Y_AXIS_MIN = 0.35
Y_AXIS_MAX = 0.85

plt.rcParams.update(
    {
        "font.size": 16,
        "font.family": "Times New Roman",
        "figure.dpi": 720,
        "axes.grid": False,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.frameon": True,
        "legend.fontsize": 16,
        "legend.facecolor": "white",
        "legend.framealpha": 0.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_summary_table(results_root: Path) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    """Load mean and sample std for each model/horizon/metric."""
    stats: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for model_key, model_label in MODELS:
        for horizon in HORIZONS:
            summary_path = results_root / model_key / f"horizon_{horizon}" / "summary.csv"
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [row for row in reader if row["fold"] != "AVG"]
            stats[(model_label, horizon)] = {}
            for metric_key, _ in METRICS:
                values = np.array([float(row[metric_key]) for row in rows], dtype=float)
                stats[(model_label, horizon)][metric_key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
    return stats


def plot_grouped_bars(stats: dict[tuple[str, int], dict[str, dict[str, float]]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    x = np.arange(len(HORIZONS))
    width = 0.12
    horizon_labels = [f"Horizon {h}" for h in HORIZONS]

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for idx, (_, model_label) in enumerate(MODELS):
            means = [stats[(model_label, horizon)][metric_key]["mean"] for horizon in HORIZONS]
            stds = [stats[(model_label, horizon)][metric_key]["std"] for horizon in HORIZONS]
            offset = (idx - (len(MODELS) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                color=MODEL_COLORS[model_label],
                alpha=0.9,
                label=model_label,
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 1, "capthick": 1},
            )
        ax.set_title(metric_label, fontsize=18, fontweight="bold")
        ax.set_xticks(x, horizon_labels)
        ax.set_ylabel("Score", fontsize=16)
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.tick_params(axis="both", labelsize=15)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=6, frameon=True, bbox_to_anchor=(0.5, 1.06), fontsize=16)
    # fig.suptitle("Model Performance by Horizon: Grouped Bars with Error Bars", fontsize=20, fontweight="bold", y=1.07)
    fig.tight_layout()
    out_path = out_dir / "table_comparison_grouped_bars.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metric_lines(stats: dict[tuple[str, int], dict[str, dict[str, float]]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for _, model_label in MODELS:
            means = [stats[(model_label, horizon)][metric_key]["mean"] for horizon in HORIZONS]
            stds = [stats[(model_label, horizon)][metric_key]["std"] for horizon in HORIZONS]
            ax.errorbar(
                HORIZONS,
                means,
                yerr=stds,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=4,
                color=MODEL_COLORS[model_label],
                label=model_label,
            )
        ax.set_title(metric_label)
        ax.set_ylabel("Score")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"Horizon {h}" for h in HORIZONS])
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Model Performance Trends Across Horizons", fontsize=15, fontweight="bold", y=1.07)
    fig.tight_layout()
    out_path = out_dir / "table_comparison_lines.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_heatmap(stats: dict[tuple[str, int], dict[str, dict[str, float]]], out_dir: Path) -> Path:
    model_labels = [label for _, label in MODELS]
    column_labels = []
    heatmap_values = []

    for model_label in model_labels:
        row = []
        for metric_key, metric_label in METRICS:
            for horizon in HORIZONS:
                row.append(stats[(model_label, horizon)][metric_key]["mean"])
                column_labels.append(f"{metric_label}\nk={horizon}")
        heatmap_values.append(row)
        column_labels = column_labels[: len(METRICS) * len(HORIZONS)]

    data = np.array(heatmap_values, dtype=float)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(data, cmap="YlGnBu", aspect="auto")
    ax.set_title("Mean Performance Heatmap Across Metrics and Horizons", fontweight="bold")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_yticklabels(model_labels)

    threshold = (float(data.max()) + float(data.min())) / 2.0
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            color = "white" if data[row, col] >= threshold else "#1f1f1f"
            ax.text(col, row, f"{data[row, col]:.3f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = out_dir / "table_comparison_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_radar(stats: dict[tuple[str, int], dict[str, dict[str, float]]], out_dir: Path) -> Path:
    metric_labels = [label for _, label in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(16, 5.5), subplot_kw={"projection": "polar"})
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, HORIZONS):
        for _, model_label in MODELS:
            values = [stats[(model_label, horizon)][metric_key]["mean"] for metric_key, _ in METRICS]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, color=MODEL_COLORS[model_label], label=model_label)
            ax.fill(angles, values, color=MODEL_COLORS[model_label], alpha=0.05)
        ax.set_title(f"Horizon {horizon}", y=1.12)
        ax.set_xticks(angles[:-1], metric_labels)
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Radar Chart Comparison by Horizon", fontsize=15, fontweight="bold", y=1.18)
    fig.tight_layout()
    out_path = out_dir / "table_comparison_radar.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary_csv(stats: dict[tuple[str, int], dict[str, dict[str, float]]], out_dir: Path) -> Path:
    out_path = out_dir / "table_comparison_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "horizon", "metric", "mean", "std"])
        for _, model_label in MODELS:
            for horizon in HORIZONS:
                for metric_key, metric_label in METRICS:
                    metric_stats = stats[(model_label, horizon)][metric_key]
                    writer.writerow(
                        [
                            model_label,
                            horizon,
                            metric_label,
                            f"{metric_stats['mean']:.6f}",
                            f"{metric_stats['std']:.6f}",
                        ]
                    )
    return out_path


def main() -> None:
    ensure_dir(REPORTS_DIR)
    stats = load_summary_table(RESULTS_ROOT)
    outputs = [
        write_summary_csv(stats, REPORTS_DIR),
        plot_grouped_bars(stats, REPORTS_DIR),
        plot_metric_lines(stats, REPORTS_DIR),
        plot_heatmap(stats, REPORTS_DIR),
        plot_radar(stats, REPORTS_DIR),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
