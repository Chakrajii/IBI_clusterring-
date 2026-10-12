#!/usr/bin/env python
"""
Run the Monte Carlo / Iterative Boltzmann Inversion schedule.

Workflow:
1. Load target RDF and initial potential.
2. Load or generate the separate random initial configuration.
3. Run the configured MC/IBI schedule.
4. Save the sampled RDF and updated potential after every update.
5. Save a run summary and run metadata.

By default, after the final IBI update an additional MC block is run so that
a final RDF sampled under the final potential is saved as
`rdf_iteration_11.csv`.

Use `--no-final-mc` if you want the strict schedule with no final MC block
after the last potential update.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.initialize import (
    load_config,
    generate_random_configuration,
    save_positions_csv,
    validate_positions,
)
from src.potential import (
    TabulatedPotential,
    initial_potential_from_target_rdf,
)
from src.rdf import (
    compute_rdf,
    save_rdf_csv,
    load_rdf_csv,
)
from src.monte_carlo import run_mc_iterations
from src.ibi import ibi_update
from src.plotting import compute_rdf_error_metrics


# =============================================================================
# Small helpers
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MC/IBI inverse-design schedule."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/parameters.yaml",
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--no-final-mc",
        action="store_true",
        help=(
            "Do not run an extra MC block after the final IBI update. "
            "With this option, the last saved RDF is the one used to compute "
            "the final potential update."
        ),
    )

    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_positions_csv(path: Path) -> np.ndarray:
    """
    Load positions from a CSV file with columns:
        particle_id,x,y
    or at least:
        x,y
    """
    df = pd.read_csv(path)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Position file {path} must contain 'x' and 'y' columns."
        )

    return df[["x", "y"]].to_numpy(dtype=float)


def ensure_target_rdf_columns(target_rdf: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the target RDF DataFrame has a `g_target` column.
    """
    target_rdf = target_rdf.copy()

    if "g_target" in target_rdf.columns:
        return target_rdf

    if "g" in target_rdf.columns:
        target_rdf["g_target"] = target_rdf["g"]
        return target_rdf

    # Fallback: assume the second column is g(r).
    if target_rdf.shape[1] >= 2:
        target_rdf["g_target"] = target_rdf.iloc[:, 1]
        return target_rdf

    raise ValueError("Could not identify target RDF values.")


def save_rdf_iteration(
    rdf_df: pd.DataFrame,
    runs_dir: Path,
    iteration: int,
) -> Path:
    """
    Save a sampled RDF as runs/rdf_iteration_XX.csv.
    """
    path = runs_dir / f"rdf_iteration_{iteration:02d}.csv"

    save_rdf_csv(
        r=rdf_df["r"].to_numpy(),
        g=rdf_df["g"].to_numpy(),
        path=path,
        value_name="g",
        counts=rdf_df["counts"].to_numpy() if "counts" in rdf_df.columns else None,
    )

    return path


def save_potential_iteration(
    potential: TabulatedPotential,
    runs_dir: Path,
    iteration: int,
) -> Path:
    """
    Save a potential as runs/potential_iteration_XX.csv.
    """
    path = runs_dir / f"potential_iteration_{iteration:02d}.csv"
    potential.save(path, value_name="u")
    return path


def run_mc_block(
    positions: np.ndarray,
    potential: TabulatedPotential,
    cfg: dict,
    n_iterations: int,
    rng: np.random.Generator,
    label: str,
):
    """
    Run one MC block and compute the sampled RDF from saved frames.
    """
    print(f"\n{label}")
    print(f"  MC iterations: {n_iterations}")

    result = run_mc_iterations(
        positions=positions,
        potential=potential,
        cfg=cfg,
        n_iterations=n_iterations,
        rng=rng,
        collect_frames=True,
        frame_interval=1,
    )

    frames = result["frames"]
    if len(frames) == 0:
        frames = [result["positions"]]

    rdf_df = compute_rdf(frames, cfg)

    print(
        f"  Acceptance ratio: {result['acceptance_ratio']:.4f} "
        f"({result['accepted_updates']}/{result['attempted_updates']})"
    )
    print(f"  Final max displacement: {result['max_displacement']:.4f}")

    return result, rdf_df


def compute_metrics_safe(
    rdf_df: pd.DataFrame,
    target_rdf: pd.DataFrame,
    cfg: dict,
) -> dict:
    """
    Compute RDF error metrics, returning NaNs on failure.
    """
    try:
        return compute_rdf_error_metrics(
            final_r=rdf_df["r"].to_numpy(),
            final_g=rdf_df["g"].to_numpy(),
            target_r=target_rdf["r"].to_numpy(),
            target_g=target_rdf["g_target"].to_numpy(),
            cfg=cfg,
        )
    except Exception as exc:
        print(f"  Warning: could not compute RDF metrics: {exc}")

        return {
            "rmse_valid_range": float("nan"),
            "rmse_positive_target": float("nan"),
            "rmse_primary": float("nan"),
            "mae_valid_range": float("nan"),
            "mae_positive_target": float("nan"),
            "mae_primary": float("nan"),
        }


# =============================================================================
# Main driver
# =============================================================================

def main():
    args = parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    data_dir = Path(paths["data_dir"])
    runs_dir = Path(paths["runs_dir"])
    results_dir = Path(paths["results_dir"])

    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load target RDF.
    # ------------------------------------------------------------------
    target_rdf_path = Path(paths["target_rdf"])

    if not target_rdf_path.exists():
        raise FileNotFoundError(
            f"Target RDF not found: {target_rdf_path}. "
            "Run run_target.py first."
        )

    target_rdf = load_rdf_csv(target_rdf_path)
    target_rdf = ensure_target_rdf_columns(target_rdf)

    print(f"Loaded target RDF: {target_rdf_path}")

    # ------------------------------------------------------------------
    # Load or create initial potential.
    # ------------------------------------------------------------------
    initial_potential_path = Path(paths["initial_potential"])

    if initial_potential_path.exists():
        potential = TabulatedPotential.from_csv(initial_potential_path, cfg)
        print(f"Loaded initial potential: {initial_potential_path}")
    else:
        print(
            "Initial potential not found. "
            "Generating it from the target RDF."
        )
        potential = initial_potential_from_target_rdf(target_rdf, cfg)
        ensure_parent_dir(paths["initial_potential"])
        potential.save(paths["initial_potential"], value_name="u_initial")

    # ------------------------------------------------------------------
    # Load or generate random initial configuration.
    # ------------------------------------------------------------------
    random_positions_path = Path(paths["random_initial_positions"])

    positions = None

    if random_positions_path.exists():
        try:
            positions = load_positions_csv(random_positions_path)
            validate_positions(positions, cfg)
            print(f"Loaded random initial positions: {random_positions_path}")
        except Exception as exc:
            print(
                f"Warning: could not use existing random positions: {exc}"
            )
            positions = None

    if positions is None:
        print("Generating random initial configuration...")
        positions = generate_random_configuration(cfg)
        save_positions_csv(positions, random_positions_path)
        print(f"Saved random initial positions: {random_positions_path}")

    # ------------------------------------------------------------------
    # MC/IBI schedule parameters.
    # ------------------------------------------------------------------
    mc_cfg = cfg["monte_carlo"]
    ibi_cfg = cfg["ibi"]

    n_updates = int(ibi_cfg["n_updates"])
    updates_per_iteration = int(mc_cfg["updates_per_iteration"])

    initial_iterations = int(mc_cfg["initial_iterations_before_first_update"])
    block_iterations = int(mc_cfg["iterations_between_updates"])

    run_final_mc = (not args.no_final_mc) and (n_updates > 0)
    final_iterations = block_iterations if run_final_mc else 0

    total_planned_iterations = (
        initial_iterations
        + max(0, n_updates - 1) * block_iterations
        + final_iterations
    )

    print("\nMC/IBI schedule:")
    print(f"  IBI updates: {n_updates}")
    print(f"  Updates per MC iteration: {updates_per_iteration}")
    print(f"  Initial MC iterations before update 1: {initial_iterations}")
    print(f"  MC iterations between later updates: {block_iterations}")
    print(f"  Final MC iterations after last update: {final_iterations}")
    print(f"  Total planned MC iterations: {total_planned_iterations}")
    print(
        "  Total planned attempted updates: "
        f"{total_planned_iterations * updates_per_iteration}"
    )

    # Random number generator for the whole MC/IBI run.
    mc_seed = mc_cfg.get("seed", cfg.get("random_seed", 0))
    rng = np.random.default_rng(mc_seed)

    # ------------------------------------------------------------------
    # Save initial potential as iteration 00.
    # ------------------------------------------------------------------
    save_potential_iteration(potential, runs_dir, 0)
    print(f"\nSaved initial potential: {runs_dir / 'potential_iteration_00.csv'}")

    summary_rows = []

    total_mc_iterations = 0
    total_attempted_updates = 0

    # ------------------------------------------------------------------
    # Handle trivial case of zero IBI updates.
    # ------------------------------------------------------------------
    if n_updates <= 0:
        print("\nNo IBI updates requested. Running initial MC block only.")

        result, rdf_df = run_mc_block(
            positions=positions,
            potential=potential,
            cfg=cfg,
            n_iterations=initial_iterations,
            rng=rng,
            label="Initial MC block",
        )

        positions = result["positions"]
        save_rdf_iteration(rdf_df, runs_dir, 0)

        metrics = compute_metrics_safe(rdf_df, target_rdf, cfg)

        summary_rows.append(
            {
                "iteration": 0,
                "ibi_update": np.nan,
                "potential_index": 0,
                "mc_iterations": result["n_iterations"],
                "attempted_updates": result["attempted_updates"],
                "accepted_updates": result["accepted_updates"],
                "acceptance_ratio": result["acceptance_ratio"],
                "max_displacement": result["max_displacement"],
                "n_updated_bins": np.nan,
                "mean_abs_correction": np.nan,
                "max_abs_correction": np.nan,
                "rdf_rmse_valid_range": metrics.get("rmse_valid_range", np.nan),
                "rdf_rmse_positive_target": metrics.get(
                    "rmse_positive_target",
                    np.nan,
                ),
                "rdf_rmse_primary": metrics.get("rmse_primary", np.nan),
            }
        )

    else:
        # --------------------------------------------------------------
        # Initial MC block using potential_iteration_00.
        # Then IBI update 1 produces potential_iteration_01.
        # --------------------------------------------------------------
        result, rdf_df = run_mc_block(
            positions=positions,
            potential=potential,
            cfg=cfg,
            n_iterations=initial_iterations,
            rng=rng,
            label="Initial MC block before IBI update 1",
        )

        positions = result["positions"]

        # Preserve adapted proposal size for subsequent blocks.
        cfg["monte_carlo"]["proposal"]["max_displacement"] = result[
            "max_displacement"
        ]

        total_mc_iterations += result["n_iterations"]
        total_attempted_updates += result["attempted_updates"]

        save_rdf_iteration(rdf_df, runs_dir, 0)

        metrics = compute_metrics_safe(rdf_df, target_rdf, cfg)

        print("\nPerforming IBI update 1...")
        potential, diagnostics = ibi_update(
            potential=potential,
            sampled_rdf=rdf_df,
            target_rdf=target_rdf,
            cfg=cfg,
        )

        save_potential_iteration(potential, runs_dir, 1)

        summary_rows.append(
            {
                "iteration": 0,
                "ibi_update": 1,
                "potential_index": 1,
                "mc_iterations": result["n_iterations"],
                "attempted_updates": result["attempted_updates"],
                "accepted_updates": result["accepted_updates"],
                "acceptance_ratio": result["acceptance_ratio"],
                "max_displacement": result["max_displacement"],
                "n_updated_bins": diagnostics["n_updated_bins"],
                "mean_abs_correction": diagnostics["mean_abs_correction"],
                "max_abs_correction": diagnostics["max_abs_correction"],
                "rdf_rmse_valid_range": metrics.get("rmse_valid_range", np.nan),
                "rdf_rmse_positive_target": metrics.get(
                    "rmse_positive_target",
                    np.nan,
                ),
                "rdf_rmse_primary": metrics.get("rmse_primary", np.nan),
            }
        )

        print(
            f"  Updated bins: {diagnostics['n_updated_bins']}"
            f"/{diagnostics['n_total_bins']}"
        )

        # --------------------------------------------------------------
        # Updates 2 through n_updates.
        # --------------------------------------------------------------
        for update_number in range(1, n_updates):
            current_potential_index = update_number
            next_potential_index = update_number + 1
            rdf_index = update_number

            result, rdf_df = run_mc_block(
                positions=positions,
                potential=potential,
                cfg=cfg,
                n_iterations=block_iterations,
                rng=rng,
                label=(
                    f"MC block before IBI update {update_number + 1} "
                    f"using potential_iteration_{current_potential_index:02d}"
                ),
            )

            positions = result["positions"]

            cfg["monte_carlo"]["proposal"]["max_displacement"] = result[
                "max_displacement"
            ]

            total_mc_iterations += result["n_iterations"]
            total_attempted_updates += result["attempted_updates"]

            save_rdf_iteration(rdf_df, runs_dir, rdf_index)

            metrics = compute_metrics_safe(rdf_df, target_rdf, cfg)

            print(f"\nPerforming IBI update {update_number + 1}...")
            potential, diagnostics = ibi_update(
                potential=potential,
                sampled_rdf=rdf_df,
                target_rdf=target_rdf,
                cfg=cfg,
            )

            save_potential_iteration(potential, runs_dir, next_potential_index)

            summary_rows.append(
                {
                    "iteration": rdf_index,
                    "ibi_update": update_number + 1,
                    "potential_index": next_potential_index,
                    "mc_iterations": result["n_iterations"],
                    "attempted_updates": result["attempted_updates"],
                    "accepted_updates": result["accepted_updates"],
                    "acceptance_ratio": result["acceptance_ratio"],
                    "max_displacement": result["max_displacement"],
                    "n_updated_bins": diagnostics["n_updated_bins"],
                    "mean_abs_correction": diagnostics["mean_abs_correction"],
                    "max_abs_correction": diagnostics["max_abs_correction"],
                    "rdf_rmse_valid_range": metrics.get(
                        "rmse_valid_range",
                        np.nan,
                    ),
                    "rdf_rmse_positive_target": metrics.get(
                        "rmse_positive_target",
                        np.nan,
                    ),
                    "rdf_rmse_primary": metrics.get("rmse_primary", np.nan),
                }
            )

            print(
                f"  Updated bins: {diagnostics['n_updated_bins']}"
                f"/{diagnostics['n_total_bins']}"
            )

        # --------------------------------------------------------------
        # Optional final MC block using the final potential.
        # This gives rdf_iteration_{n_updates}.csv.
        # --------------------------------------------------------------
        if run_final_mc:
            result, rdf_df = run_mc_block(
                positions=positions,
                potential=potential,
                cfg=cfg,
                n_iterations=final_iterations,
                rng=rng,
                label=(
                    f"Final MC block using potential_iteration_{n_updates:02d}"
                ),
            )

            positions = result["positions"]

            cfg["monte_carlo"]["proposal"]["max_displacement"] = result[
                "max_displacement"
            ]

            total_mc_iterations += result["n_iterations"]
            total_attempted_updates += result["attempted_updates"]

            save_rdf_iteration(rdf_df, runs_dir, n_updates)

            metrics = compute_metrics_safe(rdf_df, target_rdf, cfg)

            summary_rows.append(
                {
                    "iteration": n_updates,
                    "ibi_update": np.nan,
                    "potential_index": n_updates,
                    "mc_iterations": result["n_iterations"],
                    "attempted_updates": result["attempted_updates"],
                    "accepted_updates": result["accepted_updates"],
                    "acceptance_ratio": result["acceptance_ratio"],
                    "max_displacement": result["max_displacement"],
                    "n_updated_bins": np.nan,
                    "mean_abs_correction": np.nan,
                    "max_abs_correction": np.nan,
                    "rdf_rmse_valid_range": metrics.get(
                        "rmse_valid_range",
                        np.nan,
                    ),
                    "rdf_rmse_positive_target": metrics.get(
                        "rmse_positive_target",
                        np.nan,
                    ),
                    "rdf_rmse_primary": metrics.get("rmse_primary", np.nan),
                }
            )

    # ------------------------------------------------------------------
    # Save final positions for inspection.
    # ------------------------------------------------------------------
    final_positions_path = results_dir / "final_positions.csv"
    save_positions_csv(positions, final_positions_path)
    print(f"\nSaved final positions: {final_positions_path}")

    # ------------------------------------------------------------------
    # Save run summary.
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    ensure_parent_dir(paths["run_summary"])
    summary_df.to_csv(paths["run_summary"], index=False)

    print(f"Saved run summary: {paths['run_summary']}")

    # ------------------------------------------------------------------
    # Save run metadata.
    # ------------------------------------------------------------------
    metadata = pd.DataFrame(
        [
            ("timestamp", datetime.now().isoformat()),
            ("config_file", args.config),
            ("n_ibi_updates", n_updates),
            ("updates_per_mc_iteration", updates_per_iteration),
            ("initial_mc_iterations_before_first_update", initial_iterations),
            ("mc_iterations_between_updates", block_iterations),
            ("run_final_mc_after_last_update", run_final_mc),
            ("final_mc_iterations", final_iterations),
            ("total_mc_iterations", total_mc_iterations),
            ("total_attempted_updates", total_attempted_updates),
            ("mc_seed", mc_seed),
            (
                "adaptive_proposal",
                cfg["monte_carlo"].get("adaptive_proposal", True),
            ),
            (
                "final_max_displacement",
                cfg["monte_carlo"]["proposal"]["max_displacement"],
            ),
            ("target_rdf_path", str(target_rdf_path)),
            ("initial_potential_path", str(initial_potential_path)),
            ("random_initial_positions_path", str(random_positions_path)),
            ("final_positions_path", str(final_positions_path)),
        ],
        columns=["key", "value"],
    )

    metadata_path = results_dir / "run_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"Saved run metadata: {metadata_path}")

    print("\nMC/IBI run complete.")


if __name__ == "__main__":
    main()