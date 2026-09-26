"""CLI entry point.

Usage:
    python scripts/train.py --config configs/default_experiment.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experiment.run_experiment import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a cloned-agent MARL individuality experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--out", type=str, default="experiments", help="Base directory for experiment outputs.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    exp_dir, summary = run(config, base_dir=args.out)
    print("\n=== Experiment complete ===")
    print(json.dumps(summary, indent=2))
    print(f"Results written to: {exp_dir}")


if __name__ == "__main__":
    main()
