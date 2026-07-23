"""NeuralFoil solver wrapper."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def run_neuralfoil(
    coords: np.ndarray,
    alphas: list[float],
    re: float,
    model_size: str = "large",
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
) -> pd.DataFrame:
    """Run NeuralFoil for one airfoil across alphas at one Reynolds number."""
    import neuralfoil as nf

    records: list[dict] = []

    for alpha in alphas:
        try:
            result = nf.get_aero_from_coordinates(
                coordinates=coords,
                alpha=alpha,
                Re=re,
                n_crit=n_crit,
                xtr_upper=xtr_upper,
                xtr_lower=xtr_lower,
                model_size=model_size,
            )
        except Exception as exc:
            warnings.warn(
                f"NeuralFoil failed at alpha={alpha}, Re={re}: {exc}",
                stacklevel=2,
            )
            records.append(_nan_record(alpha, re))
            continue

        # Extract scalar aerodynamic coefficients
        cl = _extract_scalar(result, "CL")
        cd = _extract_scalar(result, "CD")
        cm = _extract_scalar(result, "CM")
        confidence = _extract_scalar(result, "analysis_confidence")

        # Cp_min from Cp distribution
        cp_min = _extract_cp_min(result, alpha, re)

        records.append(
            {
                "alpha": float(alpha),
                "Re": float(re),
                "Cl": cl,
                "Cd": cd,
                "Cm": cm,
                "Cp_min": cp_min,
                "analysis_confidence": confidence,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scalar(result: dict, key: str) -> float:
    """Extract a scalar float from a NeuralFoil result dict."""
    val = result.get(key)
    if val is None:
        return float("nan")
    arr = np.asarray(val)
    if arr.size == 0:
        return float("nan")
    return float(arr.flat[0])


def _extract_cp_min(result: dict, alpha: float, re: float) -> float:
    """Return min(Cp) over the surface from NeuralFoil's ue/vinf stations."""
    ue_vals: list[float] = []

    # Collect every valid ue/vinf station from both surfaces
    for surface in ("upper", "lower"):
        for i in range(32):
            key = f"{surface}_bl_ue/vinf_{i}"
            val = result.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    ue_vals.append(float(arr.flat[0]))

    if not ue_vals:
        warnings.warn(
            f"NeuralFoil returned no ue/vinf data at alpha={alpha}, "
            f"Re={re}. Setting Cp_min=NaN.",
            stacklevel=3,
        )
        return float("nan")

    ue_arr = np.array(ue_vals)
    cp_arr = 1.0 - ue_arr**2
    return float(np.nanmin(cp_arr))


def _nan_record(alpha: float, re: float) -> dict:
    """Build an all-NaN result record for a failed NeuralFoil call."""
    return {
        "alpha": float(alpha),
        "Re": float(re),
        "Cl": float("nan"),
        "Cd": float("nan"),
        "Cm": float("nan"),
        "Cp_min": float("nan"),
        "analysis_confidence": float("nan"),
    }
