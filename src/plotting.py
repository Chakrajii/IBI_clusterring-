"""
Plotting and error-metric utilities for the IBI cluster reconstruction project.

This module produces:
- final RDF vs target RDF,
- final potential vs initial target-derived potential,
- RDF absolute error and RMSE diagnostics,
- optional RMSE history plot if a run summary CSV is available.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .potential import get_cutoff_radius, get_hard_core_radius


_TOL = 1.0e-8


# =============================================================================
# Generic loaders
# =============================================================================

def _load_rdf_csv(
    path: Union[str, Path],
    preferred_y: Tuple[str, ...] = ("g", "g_target"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an RDF CSV file.

    The first column is treated as r. The y column is selected using
    preferred names if possible.
    """
    df = pd.read_csv(path)

    if "r" in df.columns:
        r = df["r"].to_numpy(dtype=float)
    else:
        r = df.iloc[:, 0].to_numpy(dtype=float)

    y_col = None

    for name in preferred_y:
        if name in df.columns:
            y_col = name
            break

    if y_col is None:
        for col in df.columns:
            if col.lower() != "r" and col.lower() != "counts":
                y_col = col
                break

    if y_col is None:
        raise ValueError(f"Could not find a g(r) column in {path}.")

    g = df[y_col].to_numpy(dtype=float)

    order = np.argsort(r)
    r = r[order]
    g = g[order]

    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    return r, g


def _load_potential_csv(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a potential CSV file.

    The first column is treated as r and the second column as u(r).
    """
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"Potential CSV {path} must contain at least two columns.")

    r = df.iloc[:, 0].to_numpy(dtype=float)
    u = df.iloc[:, 1].to_numpy(dtype=float)

    order = np.argsort(r)
    r = r[order]
    u = u[order]

    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    return r, u


def _find_latest_file(
    directory: Path,
    prefix: str,
    max_index: int,
) -> Optional[Tuple[Path, int]]:
    """
    Find the latest file of the form:
        {prefix}_iteration_{index:02d}.csv
    """
    for idx in range(max_index, -1, -1):
        path = directory / f"{prefix}_iteration_{idx:02d}.csv"
        if path.exists():
            return path, idx

    return None


# =============================================================================
# RDF error metrics
# =============================================================================

def compute_rdf_error_metrics(
    final_r: np.ndarray,
    final_g: np.ndarray,
    target_r: np.ndarray,
    target_g: np.ndarray,
    cfg: Dict,
) -> Dict:
    """
    Compute RDF error metrics on the target radial grid.

    The primary reported RMSE uses bins where the target RDF is positive
    and inside the documented valid fitting range. A broader valid-range
    RMSE is also reported.
    """
    final_r = np.asarray(final_r, dtype=float)
    final_g = np.asarray(final_g, dtype=float)
    target_r = np.asarray(target_r, dtype=float)
    target_g = np.asarray(target_g, dtype=float)

    final_g_on_target = np.interp(
        target_r,
        final_r,
        final_g,
        left=0.0,
        right=0.0,
    )

    final_g_on_target = np.nan_to_num(
        final_g_on_target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    target_g = np.nan_to_num(target_g, nan=0.0, posinf=0.0, neginf=0.0)

    diff = final_g_on_target - target_g
    abs_error = np.abs(diff)

    hard_core_radius = get_hard_core_radius(cfg)
    r_cut = get_cutoff_radius(cfg)

    ibi_cfg = cfg.get("ibi", {})
    mask_cfg = ibi_cfg.get("update_mask", {})

    min_r = float(mask_cfg.get("min_r", hard_core_radius))
    max_r = float(mask_cfg.get("max_r", r_cut))
    epsilon = float(ibi_cfg.get("epsilon", 1.0e-8))

    finite_mask = np.isfinite(final_g_on_target) & np.isfinite(target_g)

    valid_range_mask = (
        (target_r >= min_r - _TOL)
        & (target_r <= max_r + _TOL)
        & finite_mask
    )

    positive_target_mask = valid_range_mask & (target_g > epsilon)

    def _rmse(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean(diff[mask] ** 2)))

    def _mae(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(diff[mask])))

    rmse_valid_range = _rmse(valid_range_mask)
    rmse_positive_target = _rmse(positive_target_mask)

    mae_valid_range = _mae(valid_range_mask)
    mae_positive_target = _mae(positive_target_mask)

    # Choose the primary metric.
    if np.any(positive_target_mask):
        primary_mask = positive_target_mask
        rmse_primary = rmse_positive_target
        mae_primary = mae_positive_target
        primary_label = "positive-target bins"
    else:
        primary_mask = valid_range_mask
        rmse_primary = rmse_valid_range
        mae_primary = mae_valid_range
        primary_label = "valid-range bins"

    return {
        "r": target_r,
        "g_target": target_g,
        "g_final": final_g_on_target,
        "error": diff,
        "abs_error": abs_error,
        "valid_range_mask": valid_range_mask,
        "positive_target_mask": positive_target_mask,
        "primary_mask": primary_mask,
        "rmse_valid_range": rmse_valid_range,
        "rmse_positive_target": rmse_positive_target,
        "mae_valid_range": mae_valid_range,
        "mae_positive_target": mae_positive_target,
        "rmse_primary": rmse_primary,
        "mae_primary": mae_primary,
        "primary_label": primary_label,
        "min_r": min_r,
        "max_r": max_r,
        "epsilon": epsilon,
    }


# =============================================================================
# Plot helpers
# =============================================================================

def _save_figure(fig, path: Union[str, Path]) -> None:
    """
    Save a matplotlib figure and close it.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rdf_comparison(
    target_r: np.ndarray,
    target_g: np.ndarray,
    final_r: np.ndarray,
    final_g: np.ndarray,
    output_path: Union[str, Path],
    metrics: Optional[Dict] = None,
) -> None:
    """
    Plot final sampled RDF against the target RDF.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.plot(
        target_r,
        target_g,
        color="black",
        linewidth=2.0,
        label=r"target $g^*(r)$",
    )

    ax.plot(
        final_r,
        final_g,
        color="tab:blue",
        linestyle="--",
        linewidth=2.0,
        label=r"final sampled $g(r)$",
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("Final RDF vs Target RDF")

    if metrics is not None:
        rmse = metrics.get("rmse_primary", float("nan"))
        label = metrics.get("primary_label", "selected bins")

        text = f"RMSE = {rmse:.4g}\n({label})"

        ax.text(
            0.97,
            0.95,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    ax.legend()
    fig.tight_layout()

    _save_figure(fig, output_path)


def plot_potential_comparison(
    initial_r: np.ndarray,
    initial_u: np.ndarray,
    final_r: np.ndarray,
    final_u: np.ndarray,
    output_path: Union[str, Path],
    cfg: Dict,
) -> None:
    """
    Plot final IBI potential against the initial target-derived potential.

    Hard-core bins are excluded from the plot because they are usually set
    to a very large repulsive value.
    """
    hard_core_radius = get_hard_core_radius(cfg)
    r_cut = get_cutoff_radius(cfg)

    r_common = np.asarray(final_r, dtype=float)

    u_initial_common = np.interp(
        r_common,
        initial_r,
        initial_u,
        left=0.0,
        right=0.0,
    )

    u_final_common = np.asarray(final_u, dtype=float)

    mask = (
        (r_common >= hard_core_radius - _TOL)
        & (r_common <= r_cut + _TOL)
        & np.isfinite(u_initial_common)
        & np.isfinite(u_final_common)
    )

    if not np.any(mask):
        mask = np.isfinite(u_initial_common) & np.isfinite(u_final_common)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.plot(
        r_common[mask],
        u_initial_common[mask],
        color="black",
        linewidth=2.0,
        label=r"initial $u_0(r)$",
    )

    ax.plot(
        r_common[mask],
        u_final_common[mask],
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label=r"final $u_{11}(r)$",
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$u(r)$")
    ax.set_title("Final Potential vs Initial Potential")

    ax.axvline(hard_core_radius, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(r_cut, color="gray", linestyle=":", alpha=0.7)

    ax.legend()
    fig.tight_layout()

    _save_figure(fig, output_path)


def plot_rdf_error(
    target_r: np.ndarray,
    target_g: np.ndarray,
    final_r: np.ndarray,
    final_g: np.ndarray,
    output_path: Union[str, Path],
    cfg: Dict,
    metrics: Optional[Dict] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot absolute RDF error versus r.

    If a run summary DataFrame is available and contains an RMSE column,
    an RMSE-history subplot is also added.
    """
    if metrics is None:
        metrics = compute_rdf_error_metrics(
            final_r,
            final_g,
            target_r,
            target_g,
            cfg,
        )

    r = metrics["r"]
    abs_error = metrics["abs_error"]
    valid_mask = metrics["valid_range_mask"]
    rmse_primary = metrics.get("rmse_primary", float("nan"))
    primary_label = metrics.get("primary_label", "selected bins")

    rmse_col = _find_rmse_column(history_df) if history_df is not None else None
    iter_col = _find_iteration_column(history_df) if history_df is not None else None

    if history_df is not None and rmse_col is not None and iter_col is not None:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7.0, 8.0),
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
    else:
        fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
        ax2 = None

    ax1.plot(
        r,
        abs_error,
        color="tab:red",
        linewidth=1.8,
        label=r"$|g_{\mathrm{final}}(r) - g^*(r)|$",
    )

    # Shade excluded regions.
    ax1.fill_between(
        r,
        abs_error,
        where=~valid_mask,
        color="gray",
        alpha=0.15,
        label="excluded bins",
    )

    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel("Absolute RDF error")
    ax1.set_title("RDF Error")

    text = f"RMSE = {rmse_primary:.4g}\n({primary_label})"
    ax1.text(
        0.97,
        0.95,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax1.legend()

    if ax2 is not None:
        history_df = history_df.sort_values(iter_col)

        ax2.plot(
            history_df[iter_col],
            history_df[rmse_col],
            marker="o",
            color="tab:purple",
        )

        ax2.set_xlabel("IBI update")
        ax2.set_ylabel("RDF RMSE")
        ax2.set_title("RDF RMSE History")

    fig.tight_layout()

    _save_figure(fig, output_path)


# =============================================================================
# Run-summary helpers
# =============================================================================

def _find_rmse_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find an RMSE-like column in a run summary DataFrame.
    """
    if df is None:
        return None

    lower_cols = {col.lower(): col for col in df.columns}

    # Prefer positive-target RMSE if present.
    for low, original in lower_cols.items():
        if "rmse" in low and "positive" in low:
            return original

    for low, original in lower_cols.items():
        if "rmse" in low:
            return original

    return None


def _find_iteration_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find an iteration/update index column in a run summary DataFrame.
    """
    if df is None:
        return None

    lower_cols = {col.lower(): col for col in df.columns}

    for candidate in ["iteration", "ibi_update", "update", "step"]:
        if candidate in lower_cols:
            return lower_cols[candidate]

    return None


# =============================================================================
# High-level plotting driver
# =============================================================================

def make_all_plots(cfg: Dict) -> Dict:
    """
    Generate all final comparison plots using the configured file paths.

    Returns the RDF error metrics dictionary.
    """
    paths = cfg["paths"]
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(paths["runs_dir"])
    n_updates = int(cfg["ibi"]["n_updates"])

    # ------------------------------------------------------------------
    # Load target RDF.
    # ------------------------------------------------------------------
    target_rdf_path = Path(paths["target_rdf"])
    if not target_rdf_path.exists():
        raise FileNotFoundError(f"Target RDF file not found: {target_rdf_path}")

    target_r, target_g = _load_rdf_csv(
        target_rdf_path,
        preferred_y=("g_target", "g"),
    )

    # ------------------------------------------------------------------
    # Load final sampled RDF.
    # Prefer rdf_iteration_{n_updates}.csv, but fall back to latest existing.
    # ------------------------------------------------------------------
    final_rdf_info = _find_latest_file(runs_dir, "rdf", n_updates)

    if final_rdf_info is None:
        raise FileNotFoundError(
            f"No sampled RDF files found in {runs_dir}."
        )

    final_rdf_path, final_rdf_iteration = final_rdf_info
    final_r, final_g = _load_rdf_csv(
        final_rdf_path,
        preferred_y=("g", "g_sampled", "g_target"),
    )

    # ------------------------------------------------------------------
    # Load final potential.
    # ------------------------------------------------------------------
    final_potential_info = _find_latest_file(runs_dir, "potential", n_updates)

    if final_potential_info is None:
        raise FileNotFoundError(
            f"No potential files found in {runs_dir}."
        )

    final_potential_path, final_potential_iteration = final_potential_info
    final_potential_r, final_potential_u = _load_potential_csv(final_potential_path)

    # ------------------------------------------------------------------
    # Load initial potential.
    # ------------------------------------------------------------------
    initial_potential_path = Path(paths["initial_potential"])

    if not initial_potential_path.exists():
        fallback_initial = runs_dir / "potential_iteration_00.csv"
        if fallback_initial.exists():
            initial_potential_path = fallback_initial
        else:
            raise FileNotFoundError(
                "Initial potential file not found: "
                f"{paths['initial_potential']}"
            )

    initial_potential_r, initial_potential_u = _load_potential_csv(
        initial_potential_path
    )

    # ------------------------------------------------------------------
    # Compute RDF error metrics.
    # ------------------------------------------------------------------
    metrics = compute_rdf_error_metrics(
        final_r=final_r,
        final_g=final_g,
        target_r=target_r,
        target_g=target_g,
        cfg=cfg,
    )

    metrics["final_rdf_path"] = str(final_rdf_path)
    metrics["final_rdf_iteration"] = final_rdf_iteration
    metrics["final_potential_path"] = str(final_potential_path)
    metrics["final_potential_iteration"] = final_potential_iteration

    # ------------------------------------------------------------------
    # Plot RDF comparison.
    # ------------------------------------------------------------------
    plot_rdf_comparison(
        target_r=target_r,
        target_g=target_g,
        final_r=final_r,
        final_g=final_g,
        output_path=paths["rdf_comparison_plot"],
        metrics=metrics,
    )

    # ------------------------------------------------------------------
    # Plot potential comparison.
    # ------------------------------------------------------------------
    plot_potential_comparison(
        initial_r=initial_potential_r,
        initial_u=initial_potential_u,
        final_r=final_potential_r,
        final_u=final_potential_u,
        output_path=paths["potential_comparison_plot"],
        cfg=cfg,
    )

    # ------------------------------------------------------------------
    # Optional RMSE history from run summary.
    # ------------------------------------------------------------------
    history_df = None
    run_summary_path = Path(paths["run_summary"])

    if run_summary_path.exists():
        history_df = pd.read_csv(run_summary_path)

    # ------------------------------------------------------------------
    # Plot RDF error.
    # ------------------------------------------------------------------
    plot_rdf_error(
        target_r=target_r,
        target_g=target_g,
        final_r=final_r,
        final_g=final_g,
        output_path=paths["rdf_error_plot"],
        cfg=cfg,
        metrics=metrics,
        history_df=history_df,
    )

    return metrics