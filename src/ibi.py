"""
Iterative Boltzmann Inversion (IBI) for updating a tabulated pair potential.

The update used here is the stabilized damped form

    u_{n+1}(r) = u_n(r)
                 + alpha k_B T ln[ max(g_n(r), eps) / max(g*(r), eps) ]

where:
    g_n(r) is the sampled RDF,
    g*(r) is the target RDF,
    alpha is a damping factor,
    eps is a small positive floor.

The sign convention is:
    - if sampled pairs exceed the target at some r, the potential increases;
    - if sampled pairs are deficient, the potential decreases.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .potential import (
    TabulatedPotential,
    get_cutoff_radius,
    get_hard_core_radius,
    get_kbt,
)


_TOL = 1.0e-8


# =============================================================================
# RDF extraction helpers
# =============================================================================

def _interpolate_rdf_to_grid(
    r_in: np.ndarray,
    g_in: np.ndarray,
    r_grid: np.ndarray,
    counts_in: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Interpolate an RDF, and optional counts, onto the target radial grid.
    """
    r_in = np.asarray(r_in, dtype=float)
    g_in = np.asarray(g_in, dtype=float)

    if r_in.shape != g_in.shape:
        raise ValueError("r and g arrays must have the same shape.")

    order = np.argsort(r_in)
    r_sorted = r_in[order]
    g_sorted = g_in[order]

    g = np.interp(r_grid, r_sorted, g_sorted, left=0.0, right=0.0)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    counts = None
    if counts_in is not None:
        counts_in = np.asarray(counts_in, dtype=float)
        if counts_in.shape != r_in.shape:
            raise ValueError("counts array must have the same shape as r.")

        counts_sorted = counts_in[order]
        counts = np.interp(r_grid, r_sorted, counts_sorted, left=0.0, right=0.0)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

    return g, counts


def _extract_rdf_arrays(
    rdf: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    r_grid: np.ndarray,
    preferred_names: Tuple[str, ...] = ("g", "g_target"),
    counts_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract RDF values and optional counts from several possible inputs.

    Supported inputs:
        - DataFrame with columns r and g / g_target / second numeric column
        - tuple/list (r, g)
        - tuple/list (r, g, counts)
        - 1D array already aligned with r_grid
        - 2D array of shape (2, n_bins) or (3, n_bins)
    """
    r_grid = np.asarray(r_grid, dtype=float)

    # ------------------------------------------------------------------
    # DataFrame input
    # ------------------------------------------------------------------
    if isinstance(rdf, pd.DataFrame):
        df = rdf

        if "r" in df.columns:
            r_in = df["r"].to_numpy(dtype=float)

            g_col = None
            for name in preferred_names:
                if name in df.columns:
                    g_col = name
                    break

            if g_col is None:
                for col in df.columns:
                    if col != "r" and col != "counts":
                        g_col = col
                        break

            if g_col is None:
                raise ValueError("Could not find a g(r) column in RDF DataFrame.")

            g_in = df[g_col].to_numpy(dtype=float)

            counts_in = None
            if "counts" in df.columns:
                counts_in = df["counts"].to_numpy(dtype=float)

            return _interpolate_rdf_to_grid(
                r_in,
                g_in,
                r_grid,
                counts_in=counts_override if counts_override is not None else counts_in,
            )

        # DataFrame without an r column: assume values are already on r_grid.
        g_col = None
        for name in preferred_names:
            if name in df.columns:
                g_col = name
                break

        if g_col is None:
            for col in df.columns:
                if col != "counts":
                    g_col = col
                    break

        if g_col is None:
            raise ValueError("Could not find a g(r) column in RDF DataFrame.")

        g = df[g_col].to_numpy(dtype=float)
        if len(g) != len(r_grid):
            raise ValueError(
                "RDF DataFrame without an 'r' column must have length "
                "equal to the potential/RDF grid length."
            )

        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        counts = None
        if counts_override is not None:
            counts = np.asarray(counts_override, dtype=float)
        elif "counts" in df.columns:
            counts = df["counts"].to_numpy(dtype=float)

        if counts is not None:
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            if len(counts) != len(r_grid):
                raise ValueError("counts must have the same length as r_grid.")

        return g, counts

    # ------------------------------------------------------------------
    # Tuple/list input
    # ------------------------------------------------------------------
    if isinstance(rdf, (tuple, list)):
        if len(rdf) == 2:
            r_in, g_in = rdf
            counts_in = None
        elif len(rdf) == 3:
            r_in, g_in, counts_in = rdf
        else:
            raise ValueError("Tuple/list RDF input must have length 2 or 3.")

        if counts_override is not None:
            counts_in = counts_override

        return _interpolate_rdf_to_grid(r_in, g_in, r_grid, counts_in=counts_in)

    # ------------------------------------------------------------------
    # NumPy array input
    # ------------------------------------------------------------------
    arr = np.asarray(rdf, dtype=float)

    if arr.ndim == 1:
        if len(arr) != len(r_grid):
            raise ValueError(
                "1D RDF array input must already be aligned with the radial grid."
            )

        g = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        counts = None
        if counts_override is not None:
            counts = np.asarray(counts_override, dtype=float)
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            if len(counts) != len(r_grid):
                raise ValueError("counts must have the same length as r_grid.")

        return g, counts

    if arr.ndim == 2:
        if arr.shape[0] == 2:
            r_in, g_in = arr[0], arr[1]
            counts_in = counts_override
        elif arr.shape[0] == 3:
            r_in, g_in, counts_in = arr[0], arr[1], arr[2]
            if counts_override is not None:
                counts_in = counts_override
        else:
            raise ValueError(
                "2D RDF array input must have shape (2, n_bins) or (3, n_bins)."
            )

        return _interpolate_rdf_to_grid(r_in, g_in, r_grid, counts_in=counts_in)

    raise TypeError("Unsupported RDF input type.")


# =============================================================================
# Update mask
# =============================================================================

def build_update_mask(
    r: np.ndarray,
    g_target: np.ndarray,
    g_sampled: np.ndarray,
    counts: Optional[np.ndarray],
    cfg: Dict,
) -> np.ndarray:
    """
    Build a boolean mask identifying bins where IBI is allowed to update.

    Masking rules:
        - forbid hard-core bins,
        - forbid bins outside min_r/max_r,
        - forbid bins beyond the potential cutoff,
        - optionally require positive target RDF,
        - optionally require sampled counts for zero-target bins.
    """
    ibi_cfg = cfg["ibi"]
    mask_cfg = ibi_cfg.get("update_mask", {})

    hard_core_radius = get_hard_core_radius(cfg)
    r_cut = get_cutoff_radius(cfg)

    min_r = float(mask_cfg.get("min_r", hard_core_radius))
    max_r = float(mask_cfg.get("max_r", r_cut))

    require_positive_target = bool(
        mask_cfg.get("require_positive_target", True)
    )
    min_sampled_counts = float(mask_cfg.get("min_sampled_counts", 0.0))
    apply_min_counts_to_positive_target = bool(
        mask_cfg.get("apply_min_counts_to_positive_target", False)
    )

    epsilon = float(ibi_cfg.get("epsilon", 1.0e-8))

    mask = np.ones_like(r, dtype=bool)

    # Radial range and hard-core constraints.
    mask &= r >= max(hard_core_radius, min_r) - _TOL
    mask &= r <= max_r + _TOL
    mask &= r < r_cut + _TOL

    # Finite-value requirement.
    mask &= np.isfinite(g_target)
    mask &= np.isfinite(g_sampled)

    # Target-information requirement.
    if require_positive_target:
        mask &= g_target > epsilon

    # Optional sampled-count requirement.
    if counts is not None and min_sampled_counts > 0.0:
        if apply_min_counts_to_positive_target:
            mask &= counts >= min_sampled_counts - _TOL
        else:
            # For positive-target bins, allow zero sampled counts so that
            # missing sampled pairs can still drive an attractive correction.
            # For non-positive target bins, require counts if they are used.
            if not require_positive_target:
                mask &= counts >= min_sampled_counts - _TOL

    return mask


# =============================================================================
# Correction smoothing
# =============================================================================

def smooth_correction(
    r: np.ndarray,
    correction: np.ndarray,
    update_mask: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Smooth the IBI correction while keeping non-updated bins unchanged.
    """
    ibi_cfg = cfg["ibi"]
    smooth_cfg = ibi_cfg.get("correction_smoothing", {})

    enabled = bool(smooth_cfg.get("enabled", False))
    if not enabled:
        return correction

    sigma_bins = float(smooth_cfg.get("sigma_bins", 1.0))
    if sigma_bins <= 0.0:
        return correction

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return correction

    smoothed_full = gaussian_filter1d(
        correction.astype(float),
        sigma=sigma_bins,
        mode="nearest",
    )

    smoothed = np.zeros_like(correction)
    smoothed[update_mask] = smoothed_full[update_mask]

    return smoothed


# =============================================================================
# Main IBI update
# =============================================================================

def ibi_update(
    potential: TabulatedPotential,
    sampled_rdf: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    target_rdf: Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    cfg: Dict,
    sampled_counts: Optional[np.ndarray] = None,
) -> Tuple[TabulatedPotential, Dict]:
    """
    Perform one stabilized IBI potential update.

    Returns:
        new_potential, diagnostics
    """
    ibi_cfg = cfg["ibi"]

    alpha = float(ibi_cfg.get("damping_alpha", 1.0))
    epsilon = float(ibi_cfg.get("epsilon", 1.0e-8))
    max_correction = float(ibi_cfg.get("max_correction", np.inf))

    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("ibi.damping_alpha must satisfy 0 < alpha <= 1.")
    if epsilon <= 0.0:
        raise ValueError("ibi.epsilon must be positive.")

    kBT = get_kbt(cfg)

    r = potential.r.copy()
    u_old = potential.u.copy()

    hard_core_radius = get_hard_core_radius(cfg)
    hard_core_energy = float(potential.hard_core_energy)
    r_cut = get_cutoff_radius(cfg)

    # Extract sampled and target RDFs onto the potential grid.
    g_sampled, counts = _extract_rdf_arrays(
        sampled_rdf,
        r,
        preferred_names=("g", "g_sampled", "g_target"),
        counts_override=sampled_counts,
    )

    g_target, _ = _extract_rdf_arrays(
        target_rdf,
        r,
        preferred_names=("g_target", "g", "g_target"),
    )

    # Build the update mask.
    update_mask = build_update_mask(
        r=r,
        g_target=g_target,
        g_sampled=g_sampled,
        counts=counts,
        cfg=cfg,
    )

    # Regularized log ratio.
    g_sampled_safe = np.maximum(g_sampled, epsilon)
    g_target_safe = np.maximum(g_target, epsilon)

    correction = np.zeros_like(u_old)

    if np.any(update_mask):
        log_ratio = np.log(
            g_sampled_safe[update_mask] / g_target_safe[update_mask]
        )

        correction[update_mask] = alpha * kBT * log_ratio

    # Optional maximum correction magnitude.
    if np.isfinite(max_correction) and max_correction > 0.0:
        correction = np.clip(correction, -max_correction, max_correction)

    # Optional smoothing of the correction.
    correction = smooth_correction(r, correction, update_mask, cfg)

    # Apply update.
    u_new = u_old + correction

    # Preserve hard-core exclusion.
    hard_mask = r < hard_core_radius - _TOL
    u_new[hard_mask] = hard_core_energy

    # Enforce cutoff.
    cut_mask = r >= r_cut - _TOL
    u_new[cut_mask] = 0.0

    # Remove accidental non-finite values.
    u_new = np.nan_to_num(
        u_new,
        nan=0.0,
        posinf=hard_core_energy,
        neginf=-100.0,
    )

    # Re-enforce constraints after nan_to_num.
    u_new[hard_mask] = hard_core_energy
    u_new[cut_mask] = 0.0

    new_potential = TabulatedPotential(
        r,
        u_new,
        cfg,
        enforce_constraints=False,
    )

    diagnostics = {
        "r": r,
        "u_old": u_old,
        "u_new": u_new,
        "correction": correction,
        "update_mask": update_mask,
        "g_sampled": g_sampled,
        "g_target": g_target,
        "counts": counts,
        "alpha": alpha,
        "epsilon": epsilon,
        "n_updated_bins": int(np.sum(update_mask)),
        "n_total_bins": int(len(r)),
        "mean_abs_correction": float(np.mean(np.abs(correction[update_mask])))
        if np.any(update_mask)
        else 0.0,
        "max_abs_correction": float(np.max(np.abs(correction[update_mask])))
        if np.any(update_mask)
        else 0.0,
    }

    return new_potential, diagnostics