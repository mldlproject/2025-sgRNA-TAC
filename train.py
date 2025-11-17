"""Entry point for training/evaluating sgRNA-TAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sgrna_tac.config import load_config
from sgrna_tac.training.pipeline import run_pipeline
from sgrna_tac.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sgRNA-TAC model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto selects CUDA if available).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=not args.quiet)

    config = load_config(args.config)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    metrics = run_pipeline(config, device)
    print("\nTraining complete. Summary metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

