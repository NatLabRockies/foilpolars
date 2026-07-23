"""Full multi-fidelity parameter sweep orchestration."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

from foilpolars.solvers.neuralfoil_solver import run_neuralfoil
from foilpolars.solvers.xfoil_solver import run_xfoil

logger = logging.getLogger(__name__)


def n_crit_to_tu(n_crit: np.ndarray) -> np.ndarray:
    """Turbulence intensity (%) implied by n_crit, via Mack's (1977) eN fit."""
    return 100.0 * np.exp(-(n_crit + 8.43) / 2.4)


def tu_to_n_crit(tu_pct: float) -> float:
    """Inverse of n_crit_to_tu: the n_crit whose Mack-implied Tu is tu_pct."""
    return -8.43 - 2.4 * np.log(tu_pct / 100.0)


def run_full_sweep(
    config: dict,
    shapes: dict[str, np.ndarray],
    checkpoint_path: str | None = None,
    checkpoint_every: int = 250,
) -> xr.Dataset:
    """Run the sweep parallelized across foils, checkpointing every N."""
    foil_ids = list(shapes.keys())

    op_cfg = config["operating"]
    alpha_cfg = op_cfg["alpha"]
    alphas = [
        float(a)
        for a in np.arange(
            alpha_cfg["start"],
            alpha_cfg["stop"] + alpha_cfg["step"],
            alpha_cfg["step"],
        )
    ]
    re_values: list[float] = [float(r) for r in op_cfg["reynolds"]]
    n_crit_values: list[float] = [float(n) for n in op_cfg["n_crit"]]
    fidelities: list[str] = _active_fidelities(config["solvers"])
    solvers_cfg = config["solvers"]

    # Each forced-transition entry stands in for a Tu the eN method can't
    # represent (n_crit would need to go negative): it gets a sentinel
    # n_crit slot (the Mack-inverse of its Tu, for a consistent coordinate
    # label) and forces transition at xtr_top/xtr_bot instead of letting
    # eN predict it
    forced_by_ncrit: dict[float, tuple[float, float]] = {}
    for entry in op_cfg.get("forced_transition", []):
        sentinel = tu_to_n_crit(float(entry["tu"]))
        n_crit_values.append(sentinel)
        forced_by_ncrit[sentinel] = (
            float(entry["xtr_top"]), float(entry["xtr_bot"]),
        )

    # Pre-allocate NaN arrays: dims = (foil, alpha, Re, n_crit, fidelity)
    shape5 = (
        len(foil_ids), len(alphas), len(re_values), len(n_crit_values),
        len(fidelities),
    )
    nan5 = lambda: np.full(shape5, np.nan, dtype=np.float64)

    cl_arr = nan5()
    cd_arr = nan5()
    cm_arr = nan5()
    cpmin_arr = nan5()
    conv_arr = nan5()        # 1.0 / 0.0 / NaN
    conf_arr = nan5()        # NeuralFoil only

    # Build xarray Dataset. Tu is the turbulence intensity Mack's eN fit
    # implies for each n_crit; xtr_top/xtr_bot are NaN except for the
    # forced-transition sentinels, which pin transition instead of using Tu
    n_crit_arr = np.array(n_crit_values)
    xtr_top_values = [
        forced_by_ncrit[n][0] if n in forced_by_ncrit else np.nan
        for n in n_crit_values
    ]
    xtr_bot_values = [
        forced_by_ncrit[n][1] if n in forced_by_ncrit else np.nan
        for n in n_crit_values
    ]
    coords_ds = {
        "foil_id": foil_ids,
        "alpha": alphas,
        "Re": re_values,
        "n_crit": n_crit_values,
        "fidelity": fidelities,
        "Tu": ("n_crit", n_crit_to_tu(n_crit_arr)),
        "xtr_top": ("n_crit", xtr_top_values),
        "xtr_bot": ("n_crit", xtr_bot_values),
    }
    dims = ("foil_id", "alpha", "Re", "n_crit", "fidelity")
    arrays = {
        "Cl": cl_arr, "Cd": cd_arr, "Cm": cm_arr, "Cp_min": cpmin_arr,
        "converged": conv_arr, "analysis_confidence": conf_arr,
    }

    # Run one process per foil shape, capped at the core count, so the
    # independent (Re, n_crit, fidelity, alpha) sub-sweeps for different
    # shapes execute concurrently instead of one giant serial loop
    max_workers = min(len(foil_ids), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _sweep_one_foil, foil_id, shapes[foil_id], alphas,
                re_values, n_crit_values, fidelities, solvers_cfg,
                forced_by_ncrit,
            ): fi
            for fi, foil_id in enumerate(foil_ids)
        }
        # Fill in each foil's results as its process finishes (not in
        # submission order) and checkpoint the partial dataset to disk
        # every `checkpoint_every` foils, so a killed job still leaves
        # most completed foils on disk instead of only the final,
        # all-or-nothing save, without rewriting the whole file each time
        n_done = 0
        for future in as_completed(futures):
            fi = futures[future]
            cl, cd, cm, cpmin, conv, conf = future.result()
            cl_arr[fi] = cl
            cd_arr[fi] = cd
            cm_arr[fi] = cm
            cpmin_arr[fi] = cpmin
            conv_arr[fi] = conv
            conf_arr[fi] = conf
            n_done += 1
            is_last = n_done == len(foil_ids)
            if checkpoint_path is not None and (
                n_done % checkpoint_every == 0 or is_last
            ):
                _build_dataset(coords_ds, dims, arrays).to_netcdf(
                    checkpoint_path
                )
                print(f"Checkpointed {n_done}/{len(foil_ids)} foils "
                      f"to {checkpoint_path}")

    return _build_dataset(coords_ds, dims, arrays)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataset(
    coords_ds: dict, dims: tuple[str, ...], arrays: dict[str, np.ndarray],
) -> xr.Dataset:
    """Assemble the (possibly still-partial) result arrays into a Dataset."""
    ds = xr.Dataset(
        {name: (dims, arr) for name, arr in arrays.items()},
        coords=coords_ds,
    )
    ds["alpha"].attrs["units"] = "degrees"
    ds["Re"].attrs["long_name"] = "Reynolds number"
    ds["Tu"].attrs["units"] = "%"
    ds["Tu"].attrs["long_name"] = "Mack (1977) eN-implied turbulence level"
    ds["xtr_top"].attrs["long_name"] = "forced top-surface transition x/c"
    ds["xtr_bot"].attrs["long_name"] = "forced bottom-surface transition x/c"
    return ds


def _sweep_one_foil(
    foil_id: str,
    coords: np.ndarray,
    alphas: list[float],
    re_values: list[float],
    n_crit_values: list[float],
    fidelities: list[str],
    solvers_cfg: dict,
    forced_by_ncrit: dict[float, tuple[float, float]],
) -> tuple[np.ndarray, ...]:
    """Sweep every (Re, n_crit, fidelity) combination for one foil shape."""
    # Pre-allocate NaN arrays: dims = (alpha, Re, n_crit, fidelity)
    shape4 = (
        len(alphas), len(re_values), len(n_crit_values), len(fidelities),
    )
    nan4 = lambda: np.full(shape4, np.nan, dtype=np.float64)
    cl_arr, cd_arr, cm_arr, cpmin_arr = nan4(), nan4(), nan4(), nan4()
    conv_arr, conf_arr = nan4(), nan4()

    # Run every (Re, n_crit, fidelity) combination for this foil. A
    # forced-transition sentinel n_crit isn't fed to either solver as-is
    # (both mispredict for negative Ncrit); instead the solver gets a
    # nominal n_crit and the xtr_top/xtr_bot trip that actually pins
    # transition
    for ri, re in enumerate(re_values):
        for ni, n_crit in enumerate(n_crit_values):
            forced = n_crit in forced_by_ncrit
            xtr_top, xtr_bot = forced_by_ncrit.get(n_crit, (1.0, 1.0))
            solver_n_crit = 9.0 if forced else n_crit
            for fidi, fidelity in enumerate(fidelities):
                print(
                    f"  Running: foil={foil_id}  Re={re:.2e}  "
                    f"n_crit={n_crit:g}  fidelity={fidelity}"
                    + (
                        f"  xtr=({xtr_top:g}, {xtr_bot:g})" if forced else ""
                    )
                )
                df = _run_solver(
                    fidelity, coords, alphas, re, solver_n_crit,
                    solvers_cfg, xtr_top, xtr_bot,
                )
                if df is None:
                    continue

                # Fill each requested alpha's row into its slot in the
                # pre-allocated arrays, skipping any alpha the solver
                # didn't return
                for ai, alpha in enumerate(alphas):
                    row = df[np.isclose(df["alpha"], alpha)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    idx = (ai, ri, ni, fidi)
                    cl_arr[idx] = row.get("Cl", np.nan)
                    cd_arr[idx] = row.get("Cd", np.nan)
                    cm_arr[idx] = row.get("Cm", np.nan)
                    cpmin_arr[idx] = row.get("Cp_min", np.nan)

                    if fidelity == "xfoil":
                        conv_arr[idx] = row.get("converged", np.nan)
                    if fidelity == "neuralfoil":
                        conf_arr[idx] = row.get(
                            "analysis_confidence", np.nan
                        )

    return cl_arr, cd_arr, cm_arr, cpmin_arr, conv_arr, conf_arr


def _active_fidelities(solvers_cfg: dict) -> list[str]:
    """Return list of enabled fidelity labels from config."""
    order = ["neuralfoil", "xfoil"]
    return [f for f in order if solvers_cfg.get(f, {}).get("enabled", False)]


def _run_solver(
    fidelity: str,
    coords: np.ndarray,
    alphas: list[float],
    re: float,
    n_crit: float,
    solvers_cfg: dict,
    xtr_top: float = 1.0,
    xtr_bot: float = 1.0,
) -> pd.DataFrame | None:
    """Dispatch to the appropriate solver and return its DataFrame."""
    # NeuralFoil: fast surrogate, run and swallow per-alpha failures
    if fidelity == "neuralfoil":
        model_size = solvers_cfg["neuralfoil"].get("model_size", "large")
        try:
            return run_neuralfoil(
                coords, alphas, re, model_size=model_size, n_crit=n_crit,
                xtr_upper=xtr_top, xtr_lower=xtr_bot,
            )
        except Exception as exc:
            logger.error(
                "NeuralFoil sweep failed for Re=%.0f: %s", re, exc
            )
            return None

    # XFoil: viscous solver, run and swallow per-alpha failures
    if fidelity == "xfoil":
        max_iter = solvers_cfg["xfoil"].get("max_iter", 100)
        try:
            return run_xfoil(
                coords, alphas, re, n_crit=n_crit, max_iter=max_iter,
                xtr_top=xtr_top, xtr_bot=xtr_bot,
            )
        except Exception as exc:
            logger.error(
                "XFoil sweep failed for Re=%.0f: %s", re, exc
            )
            return None

    raise ValueError(f"Unknown fidelity '{fidelity}'")
