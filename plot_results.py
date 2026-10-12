#!/usr/bin/env python
"""
Generate final comparison plots from the saved MC/IBI outputs.

This script reads:
    data/target_rdf.csv
    data/potential_initial.csv
    data/runs/rdf_iteration_XX.csv
    data/runs/potential_iteration_XX.csv

and writes:
    results/rdf_comparison.png
    results/potential_comparison.png
    results/rdf_error.png
    results/rdf_metrics.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.initialize import load_config
from src.plotting import make_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate final RDF/potential comparison plots."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/parameters.yaml",
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def extract_scalar_metrics(metrics: dict) -> dict:
    """
    Keep only scalar/string metrics so they can be written to CSV.
    """
    scalar_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)):
            scalar_metrics[key] = value
        elif np.isscalar(value):
            scalar_metrics[key] = value

    return scalar_metrics


def main():
    args = parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    print("Generating final plots...")

    try:
        metrics = make_all_plots(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Error: {exc}\n\n"
            "Make sure you have already run:\n"
            "    python run_target.py\n"
            "    python run_ibi.py\n"
        )

    print("\nSaved plots:")
    print(f"  RDF comparison:       {paths['rdf_comparison_plot']}")
    print(f"  Potential comparison: {paths['potential_comparison_plot']}")
    print(f"  RDF error:            {paths['rdf_error_plot']}")

    # ------------------------------------------------------------------
    # Save scalar RDF error metrics for convenience.
    # ------------------------------------------------------------------
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "rdf_metrics.csv"
    scalar_metrics = extract_scalar_metrics(metrics)

    if scalar_metrics:
        pd.DataFrame([scalar_metrics]).to_csv(metrics_path, index=False)
        print(f"  RDF metrics:          {metrics_path}")

    # ------------------------------------------------------------------
    # Print main error metrics to stdout.
    # ------------------------------------------------------------------
    print("\nRDF error metrics:")

    if "final_rdf_path" in metrics:
        print(f"  Final RDF file: {metrics['final_rdf_path']}")

    if "final_rdf_iteration" in metrics:
        print(f"  Final RDF iteration: {metrics['final_rdf_iteration']}")

    if "rmse_primary" in metrics:
        print(f"  Primary RMSE: {metrics['rmse_primary']:.6g}")

    if "primary_label" in metrics:
        print(f"  Primary RMSE bins: {metrics['primary_label']}")

    if "rmse_valid_range" in metrics:
        print(f"  RMSE over valid range: {metrics['rmse_valid_range']:.6g}")

    if "rmse_positive_target" in metrics:
        print(
            "  RMSE over positive-target bins: "
            f"{metrics['rmse_positive_target']:.6g}"
        )

    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()