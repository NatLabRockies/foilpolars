"""Tests for shapes, solvers, and postprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# NeuralFoil solver tests
# ---------------------------------------------------------------------------

def _naca0012_coords(n: int = 100) -> np.ndarray:
    """Simple NACA 0012 via 4-digit formula for test isolation."""
    half = n // 2
    beta = np.linspace(0.0, np.pi, half)
    x = 0.5 * (1.0 - np.cos(beta))
    t = 0.12
    y_t = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    x_u, y_u = x, y_t
    x_l, y_l = x, -y_t
    xs = np.concatenate([x_l[::-1], x_u[1:]])
    ys = np.concatenate([y_l[::-1], y_u[1:]])
    return np.column_stack([xs, ys])


@pytest.mark.parametrize("alpha", [0.0])
def test_neuralfoil_returns_expected_columns(alpha: float) -> None:
    from foilpolars.solvers.neuralfoil_solver import run_neuralfoil
    coords = _naca0012_coords()
    df = run_neuralfoil(coords, alphas=[alpha], re=1e6)
    expected_cols = {"alpha", "Re", "Cl", "Cd", "Cm", "Cp_min",
                     "analysis_confidence"}
    assert expected_cols.issubset(set(df.columns))


def test_neuralfoil_cl_not_nan_at_zero() -> None:
    from foilpolars.solvers.neuralfoil_solver import run_neuralfoil
    coords = _naca0012_coords()
    df = run_neuralfoil(coords, alphas=[0.0], re=1e6)
    assert len(df) == 1
    assert np.isfinite(df.iloc[0]["Cl"]), (
        f"Cl should be finite at alpha=0; got {df.iloc[0]['Cl']}"
    )


def test_neuralfoil_multiple_alphas() -> None:
    from foilpolars.solvers.neuralfoil_solver import run_neuralfoil
    coords = _naca0012_coords()
    alphas = [-2.0, 0.0, 2.0, 4.0]
    df = run_neuralfoil(coords, alphas=alphas, re=1e6)
    assert len(df) == len(alphas)


# ---------------------------------------------------------------------------
# XFoil solver tests
# ---------------------------------------------------------------------------

def test_xfoil_returns_expected_columns() -> None:
    from foilpolars.solvers.xfoil_solver import run_xfoil
    coords = _naca0012_coords()
    df = run_xfoil(coords, alphas=[0.0], re=1e6)
    expected_cols = {"alpha", "Re", "Cl", "Cd", "Cm", "Cp_min", "converged"}
    assert expected_cols.issubset(set(df.columns))


def test_xfoil_converges_at_zero() -> None:
    from foilpolars.solvers.xfoil_solver import run_xfoil
    coords = _naca0012_coords()
    df = run_xfoil(coords, alphas=[0.0], re=1e6)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["converged"] == 1.0, "Expected convergence at alpha=0"
    assert np.isfinite(row["Cl"]), f"Cl should be finite; got {row['Cl']}"


def test_xfoil_nonconverged_not_dropped() -> None:
    """Non-converged points must remain in the DataFrame with NaN values."""
    from foilpolars.solvers.xfoil_solver import run_xfoil
    coords = _naca0012_coords()
    # Very high alpha likely won't converge; we just check the row is kept
    alphas = [0.0, 5.0, 25.0]
    df = run_xfoil(coords, alphas=alphas, re=1e6)
    assert len(df) == len(alphas), (
        "All alpha points must be in the output, including non-converged"
    )


# ---------------------------------------------------------------------------
# Postprocess tests
# ---------------------------------------------------------------------------

def _make_minimal_dataset() -> xr.Dataset:
    """Minimal Dataset for postprocess tests."""
    foils = ["16-012"]
    alphas = [0.0, 2.0, 4.0]
    re_vals = [1e6]
    n_crit_vals = [9.0]
    fids = ["neuralfoil", "xfoil"]
    shape = (1, 3, 1, 1, 2)
    dims = ("foil_id", "alpha", "Re", "n_crit", "fidelity")

    rng = np.random.default_rng(42)
    cp_data = -rng.uniform(0.1, 1.5, shape)

    return xr.Dataset(
        {
            "Cl": (dims, rng.uniform(0.0, 1.0, shape)),
            "Cd": (dims, rng.uniform(0.005, 0.02, shape)),
            "Cm": (dims, rng.uniform(-0.1, 0.0, shape)),
            "Cp_min": (dims, cp_data),
            "converged": (dims, np.ones(shape)),
            "analysis_confidence": (dims, np.ones(shape) * 0.95),
        },
        coords={
            "foil_id": foils,
            "alpha": alphas,
            "Re": re_vals,
            "n_crit": n_crit_vals,
            "fidelity": fids,
        },
    )


def test_summarize_convergence_returns_dataframe() -> None:
    from foilpolars.postprocess import summarize_convergence
    ds = _make_minimal_dataset()
    summary = summarize_convergence(ds)
    assert isinstance(summary, pd.DataFrame)
    assert "xfoil_converge_fraction" in summary.columns
    assert "neuralfoil_mean_confidence" in summary.columns
