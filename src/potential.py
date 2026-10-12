"""
Tabulated pair potential utilities.

This module handles:
- constructing the radial grid for tabulated potentials,
- initializing a potential from a target RDF using the low-density relation,
- hard-core and cutoff handling,
- potential interpolation,
- saving/loading potentials.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


_TOL = 1.0e-8


# =============================================================================
# Basic thermodynamic and geometric helpers
# =============================================================================

def get_kbt(cfg: Dict) -> float:
    """
    Return k_B T in reduced units.
    """
    system = cfg["system"]
    k_B = float(system.get("k_B", 1.0))
    T = float(system["temperature"])
    return k_B * T


def get_beta(cfg: Dict) -> float:
    """
    Return beta = 1 / (k_B T).
    """
    kBT = get_kbt(cfg)
    if kBT <= 0.0:
        raise ValueError("Temperature and k_B must give a positive k_B T.")
    return 1.0 / kBT


def get_hard_core_radius(cfg: Dict) -> float:
    """
    Return the hard-core radius / minimum allowed center-to-center distance.

    The system minimum separation is the authoritative value. If an RDF
    hard-core radius is also present, take the more conservative maximum.
    """
    system = cfg["system"]
    min_sep = float(system["min_separation"])

    rdf_cfg = cfg.get("rdf", {})
    rdf_hard_core = float(rdf_cfg.get("hard_core_radius", min_sep))

    return max(min_sep, rdf_hard_core)


# =============================================================================
# Radial grid construction
# =============================================================================

def build_r_grid(cfg: Dict) -> np.ndarray:
    """
    Build the tabulated radial grid using bin centers.

    With bin width dr and r_min = 0, the centers are:
        0.5 dr, 1.5 dr, 2.5 dr, ...
    """
    pot_cfg = cfg["potential"]

    dr = float(pot_cfg["bin_width"])
    r_min = float(pot_cfg.get("r_min", 0.0))
    r_max = float(pot_cfg["r_max"])

    if dr <= 0.0:
        raise ValueError("potential.bin_width must be positive.")
    if r_max <= r_min:
        raise ValueError("potential.r_max must be greater than potential.r_min.")

    n_bins = int(round((r_max - r_min) / dr))
    if n_bins <= 0:
        raise ValueError("Invalid radial grid: no bins found.")

    r = r_min + (np.arange(n_bins, dtype=float) + 0.5) * dr
    return r


def get_cutoff_radius(cfg: Dict) -> float:
    """
    Return the potential cutoff radius.

    If the configured cutoff is larger than the tabulated radial range,
    the tabulated range is used as the effective cutoff.
    """
    pot_cfg = cfg["potential"]
    r_cut = float(pot_cfg["r_cut"])
    r_max = float(pot_cfg["r_max"])

    return min(r_cut, r_max)


# =============================================================================
# Potential constraints: hard core, cutoff, energy reference
# =============================================================================

def apply_hard_core(
    r: np.ndarray,
    u: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Force forbidden hard-core bins to a large repulsive energy.
    """
    u = np.asarray(u, dtype=float).copy()

    hard_core_radius = get_hard_core_radius(cfg)
    hard_core_energy = float(
        cfg["potential"].get("hard_core_repulsion_energy", 1.0e6)
    )

    hard_mask = r < hard_core_radius - _TOL
    u[hard_mask] = hard_core_energy

    return u


def apply_cutoff(
    r: np.ndarray,
    u: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Set the potential to zero at and beyond the cutoff.
    """
    u = np.asarray(u, dtype=float).copy()
    r_cut = get_cutoff_radius(cfg)

    u[r >= r_cut - _TOL] = 0.0
    return u


def shift_at_cutoff(
    r: np.ndarray,
    u: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Shift the potential so that u(r_cut) = 0.

    Bins beyond the cutoff are then set to zero.
    """
    u = np.asarray(u, dtype=float).copy()
    r_cut = get_cutoff_radius(cfg)

    if len(r) == 0:
        return u

    # Interpolate the potential value at the cutoff.
    # If r_cut is outside the tabulated range, use the nearest endpoint value.
    u_cut = float(np.interp(r_cut, r, u, left=u[0], right=u[-1]))

    u -= u_cut
    u[r >= r_cut - _TOL] = 0.0

    return u


def enforce_potential_constraints(
    r: np.ndarray,
    u: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """
    Apply hard-core, cutoff, and energy-reference constraints consistently.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float).copy()

    if r.shape != u.shape:
        raise ValueError("r and u must have the same shape.")

    energy_reference = cfg["potential"].get(
        "energy_reference",
        "shift_at_cutoff",
    )

    if energy_reference == "shift_at_cutoff":
        u = shift_at_cutoff(r, u, cfg)
    else:
        u = apply_cutoff(r, u, cfg)

    # Hard-core constraints must remain enforced after shifting/cutting.
    u = apply_hard_core(r, u, cfg)

    # Make sure the cutoff is still respected after hard-core enforcement.
    u = apply_cutoff(r, u, cfg)

    return u


# =============================================================================
# Initial potential from target RDF
# =============================================================================

def _coerce_target_rdf(
    target_rdf: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """
    Accept either a CSV path or a DataFrame containing the target RDF.
    """
    if isinstance(target_rdf, (str, Path)):
        df = pd.read_csv(target_rdf)
    elif isinstance(target_rdf, pd.DataFrame):
        df = target_rdf.copy()
    else:
        raise TypeError(
            "target_rdf must be a file path or a pandas DataFrame."
        )

    if "r" not in df.columns:
        raise ValueError("target RDF must contain an 'r' column.")

    if "g_target" in df.columns:
        g_col = "g_target"
    elif "g" in df.columns:
        g_col = "g"
    elif len(df.columns) >= 2:
        g_col = df.columns[1]
    else:
        raise ValueError("target RDF must contain a target g(r) column.")

    out = pd.DataFrame(
        {
            "r": df["r"].to_numpy(dtype=float),
            "g_target": df[g_col].to_numpy(dtype=float),
        }
    )

    out = out.sort_values("r").reset_index(drop=True)
    out["g_target"] = np.nan_to_num(
        out["g_target"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return out


def initial_potential_from_target_rdf(
    target_rdf: Union[str, Path, pd.DataFrame],
    cfg: Dict,
) -> "TabulatedPotential":
    """
    Construct the initial potential using:

        u_0(r) = -k_B T ln[g*(r)]

    with numerical safeguards for zero or noisy RDF bins.
    """
    df = _coerce_target_rdf(target_rdf)

    r = build_r_grid(cfg)

    # Interpolate target RDF onto the potential grid.
    g_target = np.interp(
        r,
        df["r"].to_numpy(dtype=float),
        df["g_target"].to_numpy(dtype=float),
        left=0.0,
        right=0.0,
    )

    pot_cfg = cfg["potential"]
    low_cfg = pot_cfg.get("low_density_initialization", {})

    g_floor = float(low_cfg.get("g_floor", 1.0e-8))
    max_initial_energy = float(low_cfg.get("max_initial_energy", 100.0))
    zero_target_policy = str(low_cfg.get("zero_target_policy", "floor"))

    kBT = get_kbt(cfg)
    hard_core_radius = get_hard_core_radius(cfg)
    hard_core_energy = float(
        pot_cfg.get("hard_core_repulsion_energy", 1.0e6)
    )

    # Safeguard the logarithm.
    g_safe = np.maximum(g_target, g_floor)

    # Low-density estimate.
    u = -kBT * np.log(g_safe)

    # Optional policy:
    # - "floor": use the floored logarithm everywhere.
    # - "zero": set non-hard-core zero-target bins to zero instead of repulsion.
    if zero_target_policy == "zero":
        zero_mask = g_target <= g_floor
        allowed_mask = r >= hard_core_radius - _TOL
        u[zero_mask & allowed_mask] = 0.0

    # Clip non-hard-core energies to avoid extreme values.
    allowed_mask = r >= hard_core_radius - _TOL
    u[allowed_mask] = np.clip(
        u[allowed_mask],
        -max_initial_energy,
        max_initial_energy,
    )

    # Enforce hard core before applying cutoff/reference.
    u[r < hard_core_radius - _TOL] = hard_core_energy

    # Apply cutoff and energy reference.
    u = enforce_potential_constraints(r, u, cfg)

    return TabulatedPotential(r, u, cfg, enforce_constraints=False)


# =============================================================================
# Tabulated potential class
# =============================================================================

class TabulatedPotential:
    """
    A tabulated pair potential u(r).

    The potential is stored on bin centers and evaluated by linear
    interpolation. Hard-core and cutoff rules are enforced explicitly.
    """

    def __init__(
        self,
        r: np.ndarray,
        u: np.ndarray,
        cfg: Dict,
        enforce_constraints: bool = True,
    ):
        self.cfg = cfg

        self.r = np.asarray(r, dtype=float)
        self.u = np.asarray(u, dtype=float).copy()

        if self.r.ndim != 1 or self.u.ndim != 1:
            raise ValueError("r and u must be one-dimensional arrays.")

        if self.r.shape != self.u.shape:
            raise ValueError("r and u must have the same shape.")

        if len(self.r) == 0:
            raise ValueError("Potential grid cannot be empty.")

        if not np.all(np.diff(self.r) > 0.0):
            raise ValueError("r must be strictly increasing.")

        self.dr = float(cfg["potential"]["bin_width"])
        self.r_cut = get_cutoff_radius(cfg)
        self.hard_core_radius = get_hard_core_radius(cfg)
        self.hard_core_energy = float(
            cfg["potential"].get("hard_core_repulsion_energy", 1.0e6)
        )

        if enforce_constraints:
            self.apply_constraints()

    def apply_constraints(self) -> None:
        """
        Enforce hard-core repulsion and cutoff on the tabulated potential.
        """
        hard_mask = self.r < self.hard_core_radius - _TOL
        self.u[hard_mask] = self.hard_core_energy

        cut_mask = self.r >= self.r_cut - _TOL
        self.u[cut_mask] = 0.0

    def evaluate(self, distances: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate u(r) for one or many distances.
        """
        distances_arr = np.asarray(distances, dtype=float)
        scalar_input = distances_arr.ndim == 0

        d = np.atleast_1d(distances_arr)

        values = np.interp(
            d,
            self.r,
            self.u,
            left=self.hard_core_energy,
            right=0.0,
        )

        # Enforce hard-core and cutoff directly on evaluated distances.
        values = np.where(
            d < self.hard_core_radius - _TOL,
            self.hard_core_energy,
            values,
        )

        values = np.where(
            d >= self.r_cut - _TOL,
            0.0,
            values,
        )

        if scalar_input:
            return float(values[0])

        return values.reshape(distances_arr.shape)

    def __call__(
        self,
        distances: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Convenience call syntax: potential(r).
        """
        return self.evaluate(distances)

    def copy(self) -> "TabulatedPotential":
        """
        Return an independent copy of this potential.
        """
        return TabulatedPotential(
            self.r.copy(),
            self.u.copy(),
            self.cfg,
            enforce_constraints=False,
        )

    def save(self, path: Union[str, Path], value_name: str = "u") -> None:
        """
        Save the tabulated potential to CSV.
        """
        save_potential_csv(self.r, self.u, path, value_name=value_name)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        cfg: Dict,
    ) -> "TabulatedPotential":
        """
        Load a tabulated potential from CSV.
        """
        r, u = load_potential_csv(path)
        return cls(r, u, cfg, enforce_constraints=True)


# =============================================================================
# CSV utilities
# =============================================================================

def save_potential_csv(
    r: np.ndarray,
    u: np.ndarray,
    path: Union[str, Path],
    value_name: str = "u",
) -> None:
    """
    Save a tabulated potential to CSV.

    The output columns are:
        r,<value_name>
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)

    if r.shape != u.shape:
        raise ValueError("r and u must have the same shape.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "r": r,
            value_name: u,
        }
    )

    df.to_csv(out_path, index=False)


def load_potential_csv(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a tabulated potential from CSV.

    The first column is interpreted as r and the second column as u.
    """
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(
            "Potential CSV must contain at least two columns: r and u."
        )

    r = df.iloc[:, 0].to_numpy(dtype=float)
    u = df.iloc[:, 1].to_numpy(dtype=float)

    order = np.argsort(r)
    r = r[order]
    u = u[order]

    return r, u