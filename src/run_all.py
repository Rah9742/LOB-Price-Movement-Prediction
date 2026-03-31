"""Run multiple FI-2010 models across one or more horizons."""

import argparse
import time

from src.logistic_regression import run as run_logistic
from src.lstm_model import run as run_lstm
from src.mlp_model import run as run_mlp
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


def main():
    parser = argparse.ArgumentParser(description="Run FI-2010 LOB models")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5], choices=[1, 5, 10])
    parser.add_argument("--models", nargs="+", default=list(MODEL_RUNNERS.keys()), choices=list(MODEL_RUNNERS.keys()))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  FI-2010 LOB — FULL PIPELINE")
    print(f"  Models:   {args.models}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Debug:    {args.debug}")
    print("=" * 60 + "\n")

    for horizon in args.horizons:
        for model_name in args.models:
            print(f"\n{'#' * 60}")
            print(f"#  {model_name.upper()} | horizon k={horizon}")
            print(f"{'#' * 60}\n")
            started = time.time()
            MODEL_RUNNERS[model_name](args.data_dir, horizon, args.debug)
            print(f"\n  [{model_name}] k={horizon} completed in {time.time() - started:.1f}s\n")

    print("=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
