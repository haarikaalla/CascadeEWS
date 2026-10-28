"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .data import dataset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascadeews", description="Climate early-warning experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run the validated experiment")
    train.add_argument("--output", type=Path, default=Path("results/validated"))
    train.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    subparsers.add_parser("verify-data", help="print source checksums and row counts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "verify-data":
        print(json.dumps(dataset_manifest(ExperimentConfig()), indent=2))
        return
    from .pipeline import run_experiment, summary

    config = ExperimentConfig(output_dir=args.output, artifact_dir=args.artifacts)
    metrics = run_experiment(config)
    print(summary(metrics))


if __name__ == "__main__":
    main()
