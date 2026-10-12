#!/usr/bin/env python
"""
Generate the target configuration, target RDF, and initial potential.

By default this script also generates the separate random initial
configuration used for the inverse-design MC/IBI test.

Outputs:
    data/target_positions.csv
    data/target_rdf.csv
    data/potential_initial.csv
    data/random_initial_positions.csv   (unless --skip-random is used)
"""

import argparse
from pathlib import Path

from src.initialize import (
    load_config,
    generate_target_cluster,
    generate_random_configuration,
    save_positions_csv,
)
from src.rdf import (
    compute_target_rdf,
    save_target_rdf_csv,
)
from src.potential import (
    initial_potential_from_target_rdf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate target configuration, target RDF, initial potential, "
            "and optionally the random initial configuration."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/parameters.yaml",
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip generation of the random initial configuration.",
    )

    return parser.parse_args()


def make_parent_dirs(path: str) -> None:
    """
    Create parent directories for an output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    # ------------------------------------------------------------------
    # Prepare output directories.
    # ------------------------------------------------------------------
    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["runs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["results_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate target cluster.
    # ------------------------------------------------------------------
    print("Generating target cluster...")

    target_positions = generate_target_cluster(cfg)

    make_parent_dirs(paths["target_positions"])
    save_positions_csv(target_positions, paths["target_positions"])

    print(f"  Saved target positions: {paths['target_positions']}")
    print(f"  Number of particles: {target_positions.shape[0]}")

    # ------------------------------------------------------------------
    # 2. Compute target RDF.
    # ------------------------------------------------------------------
    print("Computing target RDF...")

    target_rdf = compute_target_rdf(target_positions, cfg)

    make_parent_dirs(paths["target_rdf"])
    save_target_rdf_csv(target_rdf, paths["target_rdf"])

    n_target_bins = len(target_rdf)
    epsilon = float(cfg.get("ibi", {}).get("epsilon", 1.0e-8))
    n_positive_target_bins = int((target_rdf["g_target"] > epsilon).sum())

    print(f"  Saved target RDF: {paths['target_rdf']}")
    print(f"  RDF bins: {n_target_bins}")
    print(f"  Positive target bins: {n_positive_target_bins}")

    # ------------------------------------------------------------------
    # 3. Compute initial potential from low-density approximation.
    # ------------------------------------------------------------------
    print("Computing initial potential from target RDF...")

    potential = initial_potential_from_target_rdf(target_rdf, cfg)

    make_parent_dirs(paths["initial_potential"])
    potential.save(paths["initial_potential"], value_name="u_initial")

    print(f"  Saved initial potential: {paths['initial_potential']}")

    allowed_mask = potential.r >= potential.hard_core_radius - 1.0e-8
    if allowed_mask.any():
        u_allowed = potential.u[allowed_mask]
        print(
            "  Initial allowed potential range: "
            f"[{u_allowed.min():.4f}, {u_allowed.max():.4f}]"
        )
    else:
        print("  Initial allowed potential range: [no allowed bins]")

    # ------------------------------------------------------------------
    # 4. Generate separate random initial configuration.
    # ------------------------------------------------------------------
    if not args.skip_random:
        print("Generating random initial configuration...")

        random_positions = generate_random_configuration(cfg)

        make_parent_dirs(paths["random_initial_positions"])
        save_positions_csv(random_positions, paths["random_initial_positions"])

        print(
            "  Saved random initial positions: "
            f"{paths['random_initial_positions']}"
        )
    else:
        print("Skipping random initial configuration generation.")

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print("\nTarget generation complete.")
    print(f"  Data directory: {paths['data_dir']}")
    print(f"  Runs directory: {paths['runs_dir']}")
    print(f"  Results directory: {paths['results_dir']}")

    print("\nReproducibility seeds:")
    print(f"  Global seed: {cfg.get('random_seed', 'not set')}")
    print(f"  Target cluster seed: {cfg['target_cluster'].get('seed', 'not set')}")
    print(
        "  Random initial configuration seed: "
        f"{cfg['random_initial_configuration'].get('seed', 'not set')}"
    )
    print(f"  Monte Carlo seed: {cfg['monte_carlo'].get('seed', 'not set')}")


if __name__ == "__main__":
    main()