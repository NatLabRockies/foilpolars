"""XFoil solver wrapper driving the locally-compiled bin/xfoil binary."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

XFOIL_BIN = Path(__file__).resolve().parents[2] / "bin" / "xfoil"
POLAR_COLS = [
    "alpha", "CL", "CD", "CDp", "CM", "Cp_min", "Top_Xtr", "Bot_Xtr",
]
RUN_TIMEOUT_S = 300


def run_xfoil(
    coords: np.ndarray,
    alphas: list[float],
    re: float,
    n_crit: float = 9.0,
    max_iter: int = 100,
    xtr_top: float = 1.0,
    xtr_bot: float = 1.0,
) -> pd.DataFrame:
    """Run XFoil for one airfoil across alphas at one Reynolds number."""
    # Sweep outward from 0 deg in two legs: ASEQ auto-halts after 4
    # consecutive non-converged points, so starting from the well-behaved
    # region protects the sweep from an early abort
    alphas_sorted = sorted(float(a) for a in alphas)

    with tempfile.TemporaryDirectory() as tmp:
        # Write the foil coordinates and batch script into the temp dir
        # XFoil will run in
        tmp_path = Path(tmp)
        _write_dat(tmp_path / "foil.dat", coords)
        script = _build_script(
            re, n_crit, max_iter, alphas_sorted, xtr_top, xtr_bot,
        )

        # Run XFoil, tolerating a hang via the timeout instead of raising
        try:
            subprocess.run(
                [str(XFOIL_BIN)],
                input=script,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("XFoil timed out for Re=%.0f", re)

        # Parse the polar file if XFoil produced one, else an empty frame
        polar_path = tmp_path / "foil.polar"
        polar = _parse_polar(polar_path) if polar_path.exists() else (
            pd.DataFrame(columns=POLAR_COLS)
        )

    return _to_result(polar, alphas_sorted, re)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dat(path: Path, coords: np.ndarray) -> None:
    """Write surface coordinates to a Selig-format .dat file."""
    lines = ["foil"]
    lines.extend(f"{x:.6f} {y:.6f}" for x, y in coords)
    path.write_text("\n".join(lines) + "\n")


def _build_script(
    re: float, n_crit: float, max_iter: int, alphas: list[float],
    xtr_top: float = 1.0, xtr_bot: float = 1.0,
) -> str:
    """Build an XFoil batch script: load, set up OPER, sweep, save polar."""
    # Derive the sweep range/step from the sorted alpha list
    a_lo, a_hi = alphas[0], alphas[-1]
    step = round(alphas[1] - alphas[0], 6) if len(alphas) > 1 else 0.5

    # XTR forces transition at a fixed x/c (top, bottom) instead of
    # letting the eN method predict it; xtr=1.0 leaves it disabled, since
    # eN can't represent bypass transition at high freestream turbulence
    return "\n".join([
        "LOAD foil.dat",
        "foil",
        "PANE",
        "OPER",
        "VPAR",
        f"N {n_crit:g}",
        f"XTR {xtr_top:g} {xtr_bot:g}",
        "",
        f"VISC {re:g}",
        "MACH 0.0",
        f"ITER {max_iter}",
        "CINC",
        "PACC",
        "foil.polar",
        "",
        f"ASEQ 0 {a_hi:g} {step:g}",
        f"ASEQ {-step:g} {a_lo:g} {-step:g}",
        "PACC",
        "",
        "QUIT",
        "",
    ])


def _parse_polar(path: Path) -> pd.DataFrame:
    """Parse an XFoil .polar file (header + dashed separator + rows)."""
    # Locate the header row, then parse the data rows beneath it
    lines = path.read_text().splitlines()
    try:
        header_idx = next(
            i for i, line in enumerate(lines)
            if line.strip().startswith("alpha")
        )
    except StopIteration:
        return pd.DataFrame(columns=POLAR_COLS)

    rows = []
    for line in lines[header_idx + 2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append([float(p) for p in parts[:len(POLAR_COLS)]])
    return pd.DataFrame(rows, columns=POLAR_COLS)


def _to_result(
    polar: pd.DataFrame, alphas: list[float], re: float,
) -> pd.DataFrame:
    """Reindex a parsed polar onto the requested alpha grid, flagging gaps."""
    # Index the parsed polar rows by their rounded alpha for lookup below
    lookup: dict[float, pd.Series] = {}
    if not polar.empty:
        for _, row in polar.iterrows():
            lookup[round(float(row["alpha"]), 2)] = row

    records = []
    for alpha in alphas:
        row = lookup.get(round(alpha, 2))
        if row is not None:
            records.append({
                "alpha": float(alpha), "Re": float(re),
                "Cl": float(row["CL"]), "Cd": float(row["CD"]),
                "Cm": float(row["CM"]), "Cp_min": float(row["Cp_min"]),
                "converged": 1.0,
            })
        else:
            # Non-convergence near stall is expected; keep the row (NaN
            # coefficients, converged=0.0) instead of dropping it
            records.append({
                "alpha": float(alpha), "Re": float(re),
                "Cl": float("nan"), "Cd": float("nan"),
                "Cm": float("nan"), "Cp_min": float("nan"),
                "converged": 0.0,
            })
    return pd.DataFrame(records)
