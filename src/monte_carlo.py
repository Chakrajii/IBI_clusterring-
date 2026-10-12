"""
Metropolis Monte Carlo for 2D particles with hard-core exclusion.

This module implements:
- random grid-compatible particle moves,
- hard-core rejection,
- pair-energy changes using a tabulated potential,
- Metropolis acceptance,
- MC iteration blocks,
- acceptance-ratio diagnostics,
- optional frame collection for RDF averaging.

Definition used in this project:
    1 MC iteration = cfg["monte_carlo"]["updates_per_iteration"] attempted updates.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .initialize import _grid_info, snap_position, validate_positions
from .rdf import pairwise_distances


_TOL = 1.0e-8


# =============================================================================
# Thermodynamic helper
# =============================================================================

def get_beta(cfg: Dict) -> float:
    """
    Return beta = 1 / (k_B T).
    """
    system = cfg["system"]
    k_B = float(system.get("k_B", 1.0))
    T = float(system["temperature"])

    kBT = k_B * T
    if kBT <= 0.0:
        raise ValueError("k_B T must be positive.")

    return 1.0 / kBT


# =============================================================================
# Safe potential evaluation
# =============================================================================

def evaluate_potential_distances(potential, distances: np.ndarray) -> np.ndarray:
    """
    Evaluate a tabulated potential on distances.

    This helper avoids interpolating from forbidden hard-core bins into
    allowed bins near the hard-core diameter.
    """
    d = np.asarray(distances, dtype=float)
    scalar_input = d.ndim == 0
    d1 = np.atleast_1d(d)

    r = np.asarray(potential.r, dtype=float)
    u = np.asarray(potential.u, dtype=float)

    hard_core_radius = float(getattr(potential, "hard_core_radius", 2.0))
    hard_core_energy = float(getattr(potential, "hard_core_energy", 1.0e6))
    r_cut = float(getattr(potential, "r_cut", np.inf))

    # Use only allowed bins for interpolation.
    allowed_mask = r >= hard_core_radius - _TOL

    if np.any(allowed_mask):
        r_allowed = r[allowed_mask]
        u_allowed = np.nan_to_num(
            u[allowed_mask],
            nan=0.0,
            posinf=hard_core_energy,
            neginf=-hard_core_energy,
        )

        # For distances just above the hard core, use the first allowed value.
        left_value = float(u_allowed[0])

        values = np.interp(
            d1,
            r_allowed,
            u_allowed,
            left=left_value,
            right=0.0,
        )
    else:
        values = np.zeros_like(d1)

    # Enforce hard core and cutoff explicitly.
    values = np.where(
        d1 < hard_core_radius - _TOL,
        hard_core_energy,
        values,
    )

    values = np.where(
        d1 >= r_cut - _TOL,
        0.0,
        values,
    )

    if scalar_input:
        return float(values[0])

    return values.reshape(d.shape)


# =============================================================================
# Energy helpers
# =============================================================================

def total_pair_energy(
    positions: np.ndarray,
    potential,
    cfg: Dict,
) -> float:
    """
    Compute the total pair energy, counting each pair once.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.shape[0] < 2:
        return 0.0

    distances = pairwise_distances(positions, cfg)
    energies = evaluate_potential_distances(potential, distances)

    return float(np.sum(energies))


def _distances_from_point(
    point: np.ndarray,
    others: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Distances from one point to a set of other points.
    """
    point = np.asarray(point, dtype=float)
    others = np.asarray(others, dtype=float)

    if others.size == 0:
        return np.array([], dtype=float)

    info = _grid_info(cfg)

    disp = others - point

    if info["periodic"]:
        disp[:, 0] -= info["Lx"] * np.round(disp[:, 0] / info["Lx"])
        disp[:, 1] -= info["Ly"] * np.round(disp[:, 1] / info["Ly"])

    return np.hypot(disp[:, 0], disp[:, 1])


# =============================================================================
# Proposal helpers
# =============================================================================

def _propose_new_position(
    positions: np.ndarray,
    particle_index: int,
    max_displacement: float,
    cfg: Dict,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Propose a new position for one particle.

    The proposal is a uniform displacement in a square, optionally snapped
    to the grid.
    """
    old = positions[particle_index]

    dx = rng.uniform(-max_displacement, max_displacement)
    dy = rng.uniform(-max_displacement, max_displacement)

    new = old + np.array([dx, dy], dtype=float)

    proposal_cfg = cfg["monte_carlo"].get("proposal", {})
    snap_to_grid = bool(proposal_cfg.get("snap_to_grid", True))

    info = _grid_info(cfg)

    if snap_to_grid:
        new = snap_position(new, cfg)
    else:
        if info["periodic"]:
            new = np.array(
                [
                    new[0] % info["Lx"],
                    new[1] % info["Ly"],
                ],
                dtype=float,
            )
        else:
            # Reject proposals outside a non-periodic box.
            if info["include_endpoints"]:
                x_ok = -_TOL <= new[0] <= info["Lx"] + _TOL
                y_ok = -_TOL <= new[1] <= info["Ly"] + _TOL
            else:
                x_ok = -_TOL <= new[0] < info["Lx"] + _TOL
                y_ok = -_TOL <= new[1] < info["Ly"] + _TOL

            if not (x_ok and y_ok):
                return None

    return new


def _same_position(p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Check whether two grid positions are the same.
    """
    return bool(np.allclose(p1, p2, rtol=0.0, atol=1.0e-12))


# =============================================================================
# Single Metropolis update
# =============================================================================

def attempt_mc_update(
    positions: np.ndarray,
    potential,
    cfg: Dict,
    rng: np.random.Generator,
    max_displacement: float,
) -> Tuple[bool, float, int, Optional[np.ndarray]]:
    """
    Attempt one Metropolis MC update.

    Returns:
        accepted: whether the move was accepted,
        delta_u: energy change (0.0 if rejected),
        particle_index: selected particle,
        new_position: proposed accepted position, or None if rejected.
    """
    n_particles = positions.shape[0]

    if n_particles <= 1:
        return False, 0.0, 0, None

    beta = get_beta(cfg)
    min_sep = float(cfg["system"]["min_separation"])

    particle_index = int(rng.integers(n_particles))
    old_position = positions[particle_index].copy()

    new_position = _propose_new_position(
        positions,
        particle_index,
        max_displacement,
        cfg,
        rng,
    )

    if new_position is None:
        return False, 0.0, particle_index, None

    # Treat identity proposals as rejected attempts.
    # This avoids artificially inflating the acceptance ratio.
    if _same_position(old_position, new_position):
        return False, 0.0, particle_index, None

    # Distances from proposed position to all other particles.
    other_mask = np.ones(n_particles, dtype=bool)
    other_mask[particle_index] = False

    others = positions[other_mask]

    new_distances = _distances_from_point(new_position, others, cfg)

    # Hard-core constraint.
    if np.any(new_distances < min_sep - _TOL):
        return False, 0.0, particle_index, None

    old_distances = _distances_from_point(old_position, others, cfg)

    old_energy = float(
        np.sum(evaluate_potential_distances(potential, old_distances))
    )
    new_energy = float(
        np.sum(evaluate_potential_distances(potential, new_distances))
    )

    delta_u = new_energy - old_energy

    if not np.isfinite(delta_u):
        return False, 0.0, particle_index, None

    if delta_u <= 0.0:
        accepted = True
    else:
        beta_du = beta * delta_u

        # Avoid overflow in exp(-beta_du).
        if beta_du > 700.0:
            accepted = False
        else:
            accepted = rng.random() < np.exp(-beta_du)

    if accepted:
        return True, float(delta_u), particle_index, new_position

    return False, 0.0, particle_index, None


# =============================================================================
# MC iteration / block driver
# =============================================================================

def run_mc_iterations(
    positions: np.ndarray,
    potential,
    cfg: Dict,
    n_iterations: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    collect_frames: bool = False,
    frame_interval: int = 1,
) -> Dict:
    """
    Run a block of MC iterations.

    One MC iteration is cfg["monte_carlo"]["updates_per_iteration"] attempted
    single-particle updates.

    If collect_frames is True, a copy of the configuration is stored after
    selected iterations for later RDF averaging.
    """
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative.")

    positions = np.asarray(positions, dtype=float).copy()

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    mc_cfg = cfg["monte_carlo"]
    updates_per_iteration = int(mc_cfg["updates_per_iteration"])

    if updates_per_iteration <= 0:
        raise ValueError("monte_carlo.updates_per_iteration must be positive.")

    if rng is None:
        if seed is None:
            seed = int(mc_cfg.get("seed", cfg.get("random_seed", 0)))
        rng = np.random.default_rng(seed)

    proposal_cfg = mc_cfg.get("proposal", {})

    max_displacement = float(proposal_cfg.get("max_displacement", 1.0))

    adaptive_proposal = bool(mc_cfg.get("adaptive_proposal", True))
    target_low, target_high = mc_cfg.get(
        "target_acceptance_range",
        [0.20, 0.60],
    )
    target_low = float(target_low)
    target_high = float(target_high)

    info = _grid_info(cfg)
    spacing = info["spacing"]

    min_displacement = max(spacing, 1.0e-12)

    default_max_limit = min(10.0, min(info["Lx"], info["Ly"]) / 4.0)
    max_displacement_limit = float(
        proposal_cfg.get("max_displacement_limit", default_max_limit)
    )
    max_displacement_limit = max(max_displacement_limit, min_displacement)

    max_displacement = min(max_displacement, max_displacement_limit)
    max_displacement = max(max_displacement, min_displacement)

    initial_total_energy = total_pair_energy(positions, potential, cfg)
    current_total_energy = initial_total_energy

    accepted_total = 0
    attempted_total = 0

    acceptance_history: List[float] = []
    max_displacement_history: List[float] = []
    frames: List[np.ndarray] = []

    for iteration in range(n_iterations):
        accepted_this_iteration = 0

        for _ in range(updates_per_iteration):
            accepted, delta_u, particle_index, new_position = attempt_mc_update(
                positions,
                potential,
                cfg,
                rng,
                max_displacement,
            )

            attempted_total += 1

            if accepted:
                accepted_total += 1
                accepted_this_iteration += 1

                positions[particle_index] = new_position
                current_total_energy += delta_u

        acceptance = (
            accepted_this_iteration / float(updates_per_iteration)
            if updates_per_iteration > 0
            else 0.0
        )

        acceptance_history.append(acceptance)
        max_displacement_history.append(max_displacement)

        if adaptive_proposal:
            if acceptance < target_low:
                max_displacement *= 0.8
            elif acceptance > target_high:
                max_displacement *= 1.2

            max_displacement = min(max_displacement, max_displacement_limit)
            max_displacement = max(max_displacement, min_displacement)

        if collect_frames and frame_interval > 0:
            if (iteration + 1) % frame_interval == 0 or iteration == n_iterations - 1:
                frames.append(positions.copy())

    overall_acceptance = (
        accepted_total / float(attempted_total)
        if attempted_total > 0
        else 0.0
    )

    # Optional final validation.
    diagnostics_cfg = cfg.get("diagnostics", {})
    if diagnostics_cfg.get("check_minimum_separation", True):
        validate_positions(positions, cfg)

    return {
        "positions": positions,
        "accepted_updates": accepted_total,
        "attempted_updates": attempted_total,
        "acceptance_ratio": overall_acceptance,
        "acceptance_history": acceptance_history,
        "max_displacement": max_displacement,
        "max_displacement_history": max_displacement_history,
        "initial_total_energy": initial_total_energy,
        "final_total_energy": current_total_energy,
        "frames": frames,
        "n_iterations": n_iterations,
    }