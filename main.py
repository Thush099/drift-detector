"""CLI entry point for streaming concept drift detection and adaptive retraining."""

from __future__ import annotations

import argparse
import logging

from config import DriftDetectorConfig
from src.pipeline import DriftDetectorPipeline
from src.utils import generate_concept_drift_stream


def build_parser() -> argparse.ArgumentParser:
    defaults = DriftDetectorConfig()
    parser = argparse.ArgumentParser(
        description="Streaming concept drift detection using ADWIN and Page-Hinkley tests."
    )
    parser.add_argument(
        "--detector",
        choices=["adwin", "page_hinkley", "both"],
        default=defaults.detector,
        help="Drift detection algorithm(s) to evaluate.",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=defaults.data_size,
        help="Total number of streaming observations to generate.",
    )
    parser.add_argument(
        "--drift-at",
        type=int,
        default=defaults.drift_at,
        help="Stream index at which concept drift begins.",
    )
    parser.add_argument(
        "--shift-magnitude",
        type=float,
        default=defaults.shift_magnitude,
        help="Target concept shift magnitude applied at drift onset.",
    )
    parser.add_argument(
        "--drift-type",
        choices=["abrupt", "gradual"],
        default=defaults.drift_type,
        help="Type of concept drift transition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed for stream generation and model initialization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    warmup_size = min(100, max(20, args.drift_at // 2 if args.drift_at > 0 else args.data_size // 5))
    X, y = generate_concept_drift_stream(
        data_size=args.data_size,
        drift_at=args.drift_at,
        shift_magnitude=args.shift_magnitude,
        drift_type=args.drift_type,
        seed=args.seed,
    )

    pipeline = DriftDetectorPipeline(
        detector=args.detector,
        warmup_size=warmup_size,
        seed=args.seed,
    )
    result = pipeline.run(X, y, drift_at=args.drift_at)
    print(result.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
