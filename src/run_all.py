"""Run multiple FI-2010 models across one or more horizons."""

import argparse
import time

from src.logistic_regression import run as run_logistic
from src.lstm_model import run as run_lstm
from src.mlp_model import run as run_mlp
from src.plot_results import (
    plot_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_heatmap,
    plot_loss_curves,
)
from src.random_forest import run as run_random_forest
from src.ridge_regression import run as run_ridge
from src.xgboost_model import run as run_xgboost


MODEL_RUNNERS = {
    "ridge": run_ridge,
    "logistic": run_logistic,
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
    "mlp": run_mlp,
    "lstm": run_lstm,
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


def generate_plots(models, horizon, results_root, out_dir, plot_type):
    """Generate reports for one horizon from saved model outputs."""
    print(f"\n  Generating {plot_type} plots for horizon k={horizon} ...")
    if plot_type in ("all", "confusion"):
        plot_confusion_matrices(models, horizon, results_root, out_dir)
    if plot_type in ("all", "loss"):
        plot_loss_curves(models, horizon, results_root, out_dir)
    if plot_type in ("all", "importance"):
        plot_feature_importance(models, horizon, results_root, out_dir)
    if plot_type in ("all", "comparison"):
        plot_comparison(models, horizon, results_root, out_dir)
    if plot_type in ("all", "heatmap"):
        plot_heatmap(models, horizon, results_root, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Run FI-2010 LOB models")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5], choices=[1, 5, 10])
    parser.add_argument("--models", nargs="+", default=list(MODEL_RUNNERS.keys()), choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument("--plot", choices=PLOT_CHOICES, help="Generate plots after training finishes for each horizon")
    parser.add_argument("--results_root", type=str, default="./output")
    parser.add_argument("--out_dir", type=str, default="./reports")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  FI-2010 LOB — FULL PIPELINE")
    print(f"  Models:   {args.models}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Debug:    {args.debug}")
    print(f"  Plot:     {args.plot or 'none'}")
    print("=" * 60 + "\n")

    for horizon in args.horizons:
        for model_name in args.models:
            print(f"\n{'#' * 60}")
            print(f"#  {model_name.upper()} | horizon k={horizon}")
            print(f"{'#' * 60}\n")
            started = time.time()
            MODEL_RUNNERS[model_name](args.data_dir, horizon, args.debug)
            print(f"\n  [{model_name}] k={horizon} completed in {time.time() - started:.1f}s\n")
        if args.plot:
            generate_plots(args.models, horizon, args.results_root, args.out_dir, args.plot)

    print("=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
