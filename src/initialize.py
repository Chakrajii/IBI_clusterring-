"""
Initialization utilities for the IBI cluster reconstruction project.

This module generates:
1. A compact target cluster of hard-core particles.
2. A separate random valid initial configuration.

Both configurations are restricted to grid points and satisfy the minimum
center-to-center separation constraint.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# Small tolerance for floating-point comparisons.
_TOL = 1.0e-8


# =============================================================================
# Configuration loading
# =============================================================================

def load_config(config_path: str = "configs/parameters.yaml") -> Dict:
    """
    Load the YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Grid and geometry helpers
# =============================================================================

def _grid_info(cfg: Dict) -> Dict:
    """
    Extract and compute grid/box information from the configuration.
    """
    system = cfg["system"]
    box = system["box"]

    Lx = float(box["Lx"])
    Ly = float(box["Ly"])
    spacing = float(system["grid_spacing"])
    include_endpoints = bool(system.get("include_endpoints", False))
    periodic = system.get("boundary_conditions", "periodic") == "periodic"

    # Number of intervals in each direction.
    n_intervals_x = int(round(Lx / spacing))
    n_intervals_y = int(round(Ly / spacing))

    # Number of grid points.
    if include_endpoints:
        nx = n_intervals_x + 1
        ny = n_intervals_y + 1
    else:
        nx = n_intervals_x
        ny = n_intervals_y

    return {
        "Lx": Lx,
        "Ly": Ly,
        "spacing": spacing,
        "include_endpoints": include_endpoints,
        "periodic": periodic,
        "n_intervals_x": n_intervals_x,
        "n_intervals_y": n_intervals_y,
        "nx": nx,
        "ny": ny,
    }


def _round_coord(value: float) -> float:
    """
    Round a coordinate to reduce floating-point representation issues.
    """
    return float(round(value, 10))


def _snap_coordinate(
    value: float,
    L: float,
    spacing: float,
    periodic: bool,
    include_endpoints: bool,
    n_intervals: int,
) -> float:
    """
    Snap a scalar coordinate to the nearest grid point.
    """
    idx = int(round(value / spacing))

    if periodic:
        # Wrap around the periodic box.
        idx %= n_intervals
    else:
        # Clamp to the allowed non-periodic grid range.
        if include_endpoints:
            max_idx = n_intervals
        else:
            max_idx = n_intervals - 1

        idx = max(0, min(idx, max_idx))

    return _round_coord(idx * spacing)


def snap_position(position: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Snap a 2D position to the nearest allowed grid point.
    """
    info = _grid_info(cfg)

    x = _snap_coordinate(
        float(position[0]),
        info["Lx"],
        info["spacing"],
        info["periodic"],
        info["include_endpoints"],
        info["n_intervals_x"],
    )
    y = _snap_coordinate(
        float(position[1]),
        info["Ly"],
        info["spacing"],
        info["periodic"],
        info["include_endpoints"],
        info["n_intervals_y"],
    )

    return np.array([x, y], dtype=float)


def _inside_box(position: np.ndarray, info: Dict) -> bool:
    """
    Check whether a position lies inside the allowed box/grid domain.
    """
    x, y = float(position[0]), float(position[1])

    Lx = info["Lx"]
    Ly = info["Ly"]
    spacing = info["spacing"]
    include_endpoints = info["include_endpoints"]

    if info["periodic"]:
        # Periodic coordinates are normally wrapped into [0, L).
        x_ok = -_TOL <= x < Lx + _TOL
        y_ok = -_TOL <= y < Ly + _TOL
        return x_ok and y_ok

    if include_endpoints:
        x_max = Lx
        y_max = Ly
    else:
        x_max = Lx - spacing
        y_max = Ly - spacing

    x_ok = -_TOL <= x <= x_max + _TOL
    y_ok = -_TOL <= y <= y_max + _TOL

    return x_ok and y_ok


def _wrap_position(position: np.ndarray, info: Dict) -> np.ndarray:
    """
    Apply periodic wrapping if required.
    """
    if not info["periodic"]:
        return position

    x = float(position[0]) % info["Lx"]
    y = float(position[1]) % info["Ly"]

    return np.array([_round_coord(x), _round_coord(y)], dtype=float)


def _minimum_image_scalar(dr: float, L: float, periodic: bool) -> float:
    """
    Apply the minimum-image convention to one scalar displacement component.
    """
    if periodic:
        dr -= L * round(dr / L)
    return dr


def _distance(p1: np.ndarray, p2: np.ndarray, cfg: Dict) -> float:
    """
    Distance between two particles using the configured boundary convention.
    """
    info = _grid_info(cfg)

    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])

    dx = _minimum_image_scalar(dx, info["Lx"], info["periodic"])
    dy = _minimum_image_scalar(dy, info["Ly"], info["periodic"])

    return float(np.hypot(dx, dy))


# =============================================================================
# Hard-core validation
# =============================================================================

def _valid_position(
    position: np.ndarray,
    existing_positions: List[np.ndarray],
    cfg: Dict,
) -> bool:
    """
    Check whether a proposed position satisfies the hard-core constraint
    with respect to all existing particles.
    """
    info = _grid_info(cfg)

    if not _inside_box(position, info):
        return False

    min_sep = float(cfg["system"]["min_separation"])

    for p in existing_positions:
        if _distance(position, p, cfg) < min_sep - _TOL:
            return False

    return True


def validate_positions(positions: np.ndarray, cfg: Dict) -> None:
    """
    Validate a full configuration.

    Checks:
    - correct number of particles,
    - positions are inside the box,
    - positions lie on the grid,
    - all pairwise separations satisfy the hard-core constraint.
    """
    info = _grid_info(cfg)
    n_expected = int(cfg["system"]["n_particles"])
    min_sep = float(cfg["system"]["min_separation"])
    spacing = info["spacing"]

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    if positions.shape[0] != n_expected:
        raise ValueError(
            f"Expected {n_expected} particles, found {positions.shape[0]}."
        )

    # Check box and grid alignment.
    for i, p in enumerate(positions):
        if not _inside_box(p, info):
            raise ValueError(f"Particle {i} is outside the allowed box.")

        gx = p[0] / spacing
        gy = p[1] / spacing

        if not np.isclose(gx, round(gx), atol=1.0e-6):
            raise ValueError(f"Particle {i} x-coordinate is not on the grid.")

        if not np.isclose(gy, round(gy), atol=1.0e-6):
            raise ValueError(f"Particle {i} y-coordinate is not on the grid.")

    # Check hard-core constraint.
    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            d = _distance(positions[i], positions[j], cfg)
            if d < min_sep - _TOL:
                raise ValueError(
                    f"Particles {i} and {j} violate the hard-core constraint: "
                    f"distance={d:.6f}, minimum={min_sep}."
                )


# =============================================================================
# Target cluster generation
# =============================================================================

def _offsets_in_shell(
    search_radius: float,
    spacing: float,
    min_radius: float,
) -> List[Tuple[float, float]]:
    """
    Generate grid offsets whose length lies between min_radius and search_radius.
    """
    steps = int(np.ceil(search_radius / spacing))

    min_radius = max(0.0, min_radius - 1.0e-8)
    min2 = min_radius * min_radius
    max2 = (search_radius + 1.0e-8) * (search_radius + 1.0e-8)

    offsets = []

    for ix in range(-steps, steps + 1):
        dx = ix * spacing
        for iy in range(-steps, steps + 1):
            dy = iy * spacing
            d2 = dx * dx + dy * dy

            # Exclude the zero offset and offsets that are too close or too far.
            if d2 <= 1.0e-12:
                continue

            if d2 < min2:
                continue

            if d2 > max2:
                continue

            offsets.append((_round_coord(dx), _round_coord(dy)))

    return offsets


def _generate_candidate_set(
    positions: List[np.ndarray],
    search_radius: float,
    min_radius: float,
    cfg: Dict,
) -> List[Tuple[float, float]]:
    """
    Generate candidate grid points near the existing cluster.
    """
    info = _grid_info(cfg)
    spacing = info["spacing"]

    offsets = _offsets_in_shell(search_radius, spacing, min_radius)
    candidates = set()

    for base in positions:
        for dx, dy in offsets:
            q = np.array([base[0] + dx, base[1] + dy], dtype=float)

            if info["periodic"]:
                q = _wrap_position(q, info)
            else:
                if not _inside_box(q, info):
                    continue

            q = np.array(
                [_round_coord(q[0]), _round_coord(q[1])],
                dtype=float,
            )

            candidates.add((float(q[0]), float(q[1])))

    # Sorting gives reproducible random selection from the candidate list.
    return sorted(candidates)


def generate_target_cluster(cfg: Dict) -> np.ndarray:
    """
    Generate the compact target cluster using a reproducible growth algorithm.
    """
    system = cfg["system"]
    target_cfg = cfg["target_cluster"]

    n_particles = int(system["n_particles"])
    info = _grid_info(cfg)

    seed = target_cfg.get("seed", cfg.get("random_seed", 0))
    rng = np.random.default_rng(seed)

    center = target_cfg.get(
        "initial_center",
        [info["Lx"] / 2.0, info["Ly"] / 2.0],
    )

    first_position = snap_position(np.array(center, dtype=float), cfg)
    positions: List[np.ndarray] = [first_position]

    candidate_cfg = target_cfg["candidate_generation"]
    base_search_radius = float(candidate_cfg["max_distance_to_existing"])
    min_candidate_radius = float(candidate_cfg["min_distance_to_existing"])
    max_attempts_per_particle = int(
        candidate_cfg.get("max_attempts_per_particle", 200000)
    )

    # Maximum radius to try if the local candidate shell is empty.
    max_search_radius = 0.5 * min(info["Lx"], info["Ly"])

    while len(positions) < n_particles:
        valid_candidates = []

        search_radius = base_search_radius
        while search_radius <= max_search_radius + _TOL:
            candidates = _generate_candidate_set(
                positions,
                search_radius,
                min_candidate_radius,
                cfg,
            )

            valid_candidates = []
            for c in candidates:
                c_arr = np.array(c, dtype=float)
                if _valid_position(c_arr, positions, cfg):
                    valid_candidates.append(c)

            if valid_candidates:
                break

            search_radius += info["spacing"]

        if valid_candidates:
            idx = rng.integers(len(valid_candidates))
            chosen = np.array(valid_candidates[idx], dtype=float)
            positions.append(chosen)
        else:
            # Fallback: rejection-sampled random grid point.
            found = False

            for _ in range(max_attempts_per_particle):
                ix = rng.integers(info["nx"])
                iy = rng.integers(info["ny"])

                p = np.array(
                    [ix * info["spacing"], iy * info["spacing"]],
                    dtype=float,
                )

                if _valid_position(p, positions, cfg):
                    positions.append(p)
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    "Could not place all target particles. "
                    "Try increasing candidate search radius or max attempts."
                )

    positions = np.vstack(positions).reshape(n_particles, 2)
    validate_positions(positions, cfg)

    return positions


# =============================================================================
# Random initial configuration generation
# =============================================================================

def generate_random_configuration(cfg: Dict) -> np.ndarray:
    """
    Generate a random valid configuration of particles on the grid.

    This configuration is used as the starting point for the inverse-design
    MC/IBI test. It must not be the target cluster.
    """
    system = cfg["system"]
    random_cfg = cfg["random_initial_configuration"]

    n_particles = int(system["n_particles"])
    min_sep = float(system["min_separation"])

    info = _grid_info(cfg)

    seed = random_cfg.get("seed", cfg.get("random_seed", 0))
    rng = np.random.default_rng(seed)

    max_attempts_per_particle = int(
        random_cfg.get("max_attempts_per_particle", 500000)
    )

    positions: List[np.ndarray] = []

    for particle_idx in range(n_particles):
        placed = False

        for _ in range(max_attempts_per_particle):
            ix = rng.integers(info["nx"])
            iy = rng.integers(info["ny"])

            p = np.array(
                [ix * info["spacing"], iy * info["spacing"]],
                dtype=float,
            )

            if _valid_position(p, positions, cfg):
                positions.append(p)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place random particle {particle_idx} after "
                f"{max_attempts_per_particle} attempts."
            )

    positions = np.vstack(positions).reshape(n_particles, 2)
    validate_positions(positions, cfg)

    return positions


# =============================================================================
# Saving utilities
# =============================================================================

def save_positions_csv(positions: np.ndarray, path: str) -> None:
    """
    Save particle positions to CSV with columns:
        particle_id,x,y
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "particle_id": np.arange(positions.shape[0]),
            "x": positions[:, 0],
            "y": positions[:, 1],
        }
    )

    df.to_csv(out_path, index=False)