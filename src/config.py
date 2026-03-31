"""Shared CLI configuration for FI-2010 model scripts."""

import argparse


def parse_args(description: str = "FI-2010 LOB Model"):
    """Parse model CLI arguments shared across all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the FI-2010 data directory")
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10], help="Prediction horizon k")
    parser.add_argument("--debug", action="store_true", help="Use truncated data and smaller search/training settings")
    return parser.parse_args()
