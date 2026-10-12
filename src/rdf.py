"""
Radial distribution function (RDF) utilities for a 2D particle system.

This module implements a 2D RDF estimator using unique particle pairs i < j.

For a homogeneous 2D system, the estimator is

    g(r) = 2 H(r) / [N rho Delta A(r)]

where:
    H(r) is the number of unique pairs in the bin,
    N is the number of particles,
    rho = N / A is the number density,
    Delta A(r) is the annular shell area.

The same estimator and density convention are used for:
    - the target configuration,
    - random initial configurations,
    - MC-sampled configurations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


_TOL = 1.0e-8


# =============================================================================
# RDF grid
# =============================================================================

def build_rdf_grid(cfg: Dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build RDF bin centers and bin edges.

    Returns:
        r: bin centers, shape (n_bins,)
        edges: bin edges, shape (n_bins + 1,)
    """
    rdf_cfg = cfg["rdf"]

    dr = float(rdf_cfg["bin_width"])
    r_min = float(rdf_cfg.get("r_min", 0.0))
    r_max = float(rdf_cfg["r_max"])

    if dr <= 0.0:
        raise ValueError("rdf.bin_width must be positive.")
    if r_max <= r_min:
        raise ValueError("rdf.r_max must be greater than rdf.r_min.")

    n_bins = int(round((r_max - r_min) / dr))
    if n_bins <= 0:
        raise ValueError("Invalid RDF grid: no bins found.")

    edges = r_min + np.arange(n_bins + 1, dtype=float) * dr
    r = r_min + (np.arange(n_bins, dtype=float) + 0.5) * dr

    return r, edges


def shell_areas_2d(edges: np.ndarray) -> np.ndarray:
    """
    Compute exact 2D annular shell areas from bin edges.

    For bin edges [r_k, r_{k+1}], the area is

        Delta A_k = pi (r_{k+1}^2 - r_k^2)
    """
    edges = np.asarray(edges, dtype=float)
    return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


# =============================================================================
# Geometry helpers
# =============================================================================

def _system_info(cfg: Dict) -> Dict:
    """
    Extract basic system information needed for distance calculations.
    """
    system = cfg["system"]
    box = system["box"]

    return {
        "Lx": float(box["Lx"]),
        "Ly": float(box["Ly"]),
        "periodic": system.get("boundary_conditions", "periodic") == "periodic",
        "min_separation": float(system["min_separation"]),
    }


def _hard_core_radius(cfg: Dict) -> float:
    """
    Return the hard-core radius / minimum allowed separation.
    """
    system = cfg["system"]
    min_sep = float(system["min_separation"])

    rdf_cfg = cfg.get("rdf", {})
    rdf_hard_core = float(rdf_cfg.get("hard_core_radius", min_sep))

    return max(min_sep, rdf_hard_core)


def pairwise_distances(positions: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Compute all unique pair distances i < j.

    Uses the minimum-image convention if periodic boundaries are enabled.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    n = positions.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    info = _system_info(cfg)
    Lx = info["Lx"]
    Ly = info["Ly"]
    periodic = info["periodic"]

    distances = []

    for i in range(n - 1):
        dr = positions[i + 1:] - positions[i]

        if periodic:
            dr[:, 0] -= Lx * np.round(dr[:, 0] / Lx)
            dr[:, 1] -= Ly * np.round(dr[:, 1] / Ly)

        d = np.hypot(dr[:, 0], dr[:, 1])
        distances.append(d)

    if not distances:
        return np.array([], dtype=float)

    return np.concatenate(distances)


# =============================================================================
# Density normalization
# =============================================================================

def number_density(n_particles: int, cfg: Dict) -> float:
    """
    Return the number density used for RDF normalization.

    Supported modes:
        "global_box": rho = N / box_area
        "unit":       rho = 1, mainly for debugging
    """
    rdf_cfg = cfg.get("rdf", {})
    mode = str(rdf_cfg.get("density_mode", "global_box"))

    if mode == "global_box":
        info = _system_info(cfg)
        area = info["Lx"] * info["Ly"]
        if area <= 0.0:
            raise ValueError("Box area must be positive.")
        return float(n_particles) / area

    if mode == "unit":
        return 1.0

    raise ValueError(
        f"Unknown rdf.density_mode: {mode}. "
        "Supported values are 'global_box' and 'unit'."
    )


# =============================================================================
# RDF computation
# =============================================================================

def _normalize_frames(
    positions: Union[np.ndarray, Sequence[np.ndarray]],
) -> List[np.ndarray]:
    """
    Accept either a single configuration or a sequence of configurations.

    Returns a list of arrays, each with shape (N, 2).
    """
    if isinstance(positions, np.ndarray):
        if positions.ndim == 2:
            return [positions]
        if positions.ndim == 3:
            return [positions[i] for i in range(positions.shape[0])]

    if isinstance(positions, (list, tuple)):
        frames = []
        for frame in positions:
            arr = np.asarray(frame, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Each frame must have shape (N, 2).")
            frames.append(arr)
        return frames

    raise TypeError(
        "positions must be an array of shape (N, 2), "
        "an array of shape (n_frames, N, 2), "
        "or a list of (N, 2) arrays."
    )


def compute_rdf(
    positions: Union[np.ndarray, Sequence[np.ndarray]],
    cfg: Dict,
    smooth: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute the RDF for one configuration or average over several frames.

    Returns a DataFrame with columns:
        r, g, counts

    `counts` is the average number of unique pairs per frame in each bin.
    """
    frames = _normalize_frames(positions)

    if len(frames) == 0:
        raise ValueError("At least one configuration frame is required.")

    r, edges = build_rdf_grid(cfg)
    areas = shell_areas_2d(edges)

    n_bins = len(r)

    g_sum = np.zeros(n_bins, dtype=float)
    counts_sum = np.zeros(n_bins, dtype=float)

    n_valid_frames = 0

    for frame in frames:
        n_particles = frame.shape[0]

        if n_particles < 2:
            continue

        distances = pairwise_distances(frame, cfg)

        hist, _ = np.histogram(distances, bins=edges)
        hist = hist.astype(float)

        rho = number_density(n_particles, cfg)

        denominator = n_particles * rho * areas

        with np.errstate(divide="ignore", invalid="ignore"):
            g_frame = 2.0 * hist / denominator

        g_frame = np.nan_to_num(g_frame, nan=0.0, posinf=0.0, neginf=0.0)

        g_sum += g_frame
        counts_sum += hist
        n_valid_frames += 1

    if n_valid_frames == 0:
        g = np.zeros_like(r)
        counts = np.zeros_like(r)
    else:
        g = g_sum / float(n_valid_frames)
        counts = counts_sum / float(n_valid_frames)

    # Enforce exact zeros in the forbidden hard-core region.
    hard_core_radius = _hard_core_radius(cfg)
    g[r < hard_core_radius - _TOL] = 0.0
    counts[r < hard_core_radius - _TOL] = 0.0

    # Optional smoothing.
    if smooth is None:
        smooth = bool(cfg.get("rdf", {}).get("smoothing", {}).get("enabled", False))

    if smooth:
        g = smooth_rdf(r, g, cfg)

    # Ensure non-negative RDF values after smoothing.
    g = np.maximum(g, 0.0)

    return pd.DataFrame(
        {
            "r": r,
            "g": g,
            "counts": counts,
        }
    )


# =============================================================================
# Smoothing
# =============================================================================

def smooth_rdf(
    r: np.ndarray,
    g: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Smooth the RDF using a Gaussian filter over bins.

    The hard-core region is preserved as zero.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float).copy()

    rdf_cfg = cfg.get("rdf", {})
    smooth_cfg = rdf_cfg.get("smoothing", {})

    enabled = bool(smooth_cfg.get("enabled", False))
    if not enabled:
        return g

    sigma_bins = float(smooth_cfg.get("sigma_bins", 1.0))
    if sigma_bins <= 0.0:
        return g

    hard_core_radius = _hard_core_radius(cfg)
    min_r_for_smoothing = float(
        smooth_cfg.get("min_r_for_smoothing", hard_core_radius)
    )

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        # If SciPy is unavailable, return the unsmoothed RDF.
        return g

    smoothed_full = gaussian_filter1d(
        g.astype(float),
        sigma=sigma_bins,
        mode="nearest",
    )

    smooth_mask = r >= min_r_for_smoothing - _TOL
    g[smooth_mask] = smoothed_full[smooth_mask]

    # Hard-core bins must remain zero.
    g[r < hard_core_radius - _TOL] = 0.0

    # Avoid small negative values caused by numerical smoothing.
    g = np.maximum(g, 0.0)

    return g


# =============================================================================
# Target RDF convenience
# =============================================================================

def compute_target_rdf(
    positions: np.ndarray,
    cfg: Dict,
    smooth: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute the target RDF and return columns:
        r, g_target
    """
    df = compute_rdf(positions, cfg, smooth=smooth)

    target_df = pd.DataFrame(
        {
            "r": df["r"],
            "g_target": df["g"],
        }
    )

    return target_df


# =============================================================================
# Floor / regularization helpers
# =============================================================================

def floor_rdf_values(
    g: Union[np.ndarray, pd.Series],
    epsilon: float,
) -> np.ndarray:
    """
    Replace RDF values below epsilon with epsilon.

    This is useful before taking logarithms in IBI.
    """
    g_arr = np.asarray(g, dtype=float)

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    return np.maximum(g_arr, epsilon)


# =============================================================================
# Saving / loading RDF files
# =============================================================================

def save_rdf_csv(
    r: np.ndarray,
    g: np.ndarray,
    path: Union[str, Path],
    value_name: str = "g",
    counts: Optional[np.ndarray] = None,
) -> None:
    """
    Save an RDF to CSV.

    Default columns:
        r,g

    If counts is provided:
        r,g,counts
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    if r.shape != g.shape:
        raise ValueError("r and g must have the same shape.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "r": r,
        value_name: g,
    }

    if counts is not None:
        counts = np.asarray(counts, dtype=float)
        if counts.shape != r.shape:
            raise ValueError("counts must have the same shape as r.")
        data["counts"] = counts

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)


def save_target_rdf_csv(
    target_rdf: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """
    Save target RDF in the required format:
        r,g_target
    """
    if "r" not in target_rdf.columns or "g_target" not in target_rdf.columns:
        raise ValueError("target_rdf must contain columns 'r' and 'g_target'.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_rdf[["r", "g_target"]].to_csv(out_path, index=False)


def load_rdf_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load an RDF CSV file.
    """
    return pd.read_csv(path)