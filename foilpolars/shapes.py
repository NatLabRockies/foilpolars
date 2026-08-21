"""Airfoil coordinate loaders using AeroSandbox."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from foilpolars.utils import save_or_show

if TYPE_CHECKING:
    import aerosandbox as asb

FIXES_DIR = os.path.join(os.path.dirname(__file__), "airfoil_fixes")

# Sandia's MHKF1 hydrofoil family, not in AeroSandbox's UIUC-backed
# database; vendored from github.com/codykarcher/metafoil (MIT licensed)
MHK_DIR = os.path.join(os.path.dirname(__file__), "mhk_airfoils")

# Rest of metafoil's raw_airfoil_files database (~6.5k airfoils, same
# source/license as MHK_DIR), vendored so its designations resolve too
METAFOIL_DIR = os.path.join(os.path.dirname(__file__), "metafoil_airfoils")

# Local coordinate directories checked before falling back to AeroSandbox's
# UIUC lookup, in priority order
_LOCAL_DIRS = [FIXES_DIR, MHK_DIR, METAFOIL_DIR]


def _load_airfoil(desig: str) -> asb.Airfoil:
    """Build an aerosandbox Airfoil, preferring a local coordinate file."""
    import aerosandbox as asb

    # Some designations aren't in (or are known wrong in) AeroSandbox's
    # UIUC-backed database; a local coordinate file overrides/supplies it
    for local_dir in _LOCAL_DIRS:
        local_path = os.path.join(local_dir, f"{desig}.dat")
        if os.path.exists(local_path):
            coordinates = np.loadtxt(local_path)
            return asb.Airfoil(name=desig, coordinates=coordinates)

    return asb.Airfoil(desig)


def load_all_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load all airfoil coordinates, repanelling per the config's settings."""
    repanel_cfg = config.get("repanel", {})
    enabled = repanel_cfg.get("enabled", True)
    n_points_per_side = repanel_cfg.get("n_points_per_side", 100)

    shapes: dict[str, np.ndarray] = {}

    # Load coordinates for every airfoil designation
    for desig in config["shapes"]:
        af = _load_airfoil(desig)
        if enabled:
            af = af.repanel(n_points_per_side=n_points_per_side)
        shapes[desig] = af.coordinates

    return shapes


def load_raw_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load airfoil coordinates as published, without repaneling."""
    shapes: dict[str, np.ndarray] = {}

    # Load each airfoil's coordinates straight from its source table
    for desig in config["shapes"]:
        af = _load_airfoil(desig)
        shapes[desig] = af.coordinates

    return shapes


def save_shapes(
    shapes: dict[str, np.ndarray],
    out_path: str = "output/data/shapes.nc",
    columns: tuple[str, str] = ("x", "y"),
    extra_columns: dict[str, dict[str, object]] | None = None,
) -> None:
    """Write every foil's coordinates + metadata to one netcdf file."""
    import xarray as xr

    # netCDF forbids "/" in variable names, so chord-normalized x/c, y/c
    # coordinates are stored as plain "x", "y" by default
    foil_ids = list(shapes.keys())
    n_points = np.array([shapes[fid].shape[0] for fid in foil_ids])

    # Foils can have different point counts (e.g. raw UIUC tables), so
    # pad to the longest with NaN; n_points records true length for reload
    max_n = int(n_points.max())
    coords = np.full((len(foil_ids), max_n, 2), np.nan)
    for i, foil_id in enumerate(foil_ids):
        pts = shapes[foil_id]
        coords[i, :pts.shape[0], :] = pts

    data_vars = {
        columns[0]: (("foil_id", "point"), coords[:, :, 0]),
        columns[1]: (("foil_id", "point"), coords[:, :, 1]),
        "n_points": (("foil_id",), n_points),
    }

    # Caller-supplied per-foil metadata (e.g. affine transform, PGA
    # coords) merged in as extra data variables
    if extra_columns is not None:
        extra_names = list(next(iter(extra_columns.values())).keys())
        for name in extra_names:
            data_vars[name] = (
                ("foil_id",),
                [extra_columns[fid][name] for fid in foil_ids],
            )

    ds = xr.Dataset(data_vars, coords={"foil_id": foil_ids})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved {len(foil_ids)} shapes to {out_path}")


def load_shapes(
    path: str,
    columns: tuple[str, str] = ("x", "y"),
) -> dict[str, np.ndarray]:
    """Read a save_shapes netcdf back into a foil_id -> coordinates dict."""
    import xarray as xr

    ds = xr.load_dataset(path)
    shapes: dict[str, np.ndarray] = {}

    # Trim each foil's NaN padding back off using its stored n_points
    for foil_id in ds["foil_id"].values:
        n = int(ds["n_points"].sel(foil_id=foil_id).item())
        row = ds.sel(foil_id=foil_id)
        shapes[str(foil_id)] = np.column_stack(
            [row[columns[0]].values[:n], row[columns[1]].values[:n]],
        )
    return shapes


def plot_shapes(
    shapes: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Overlay all airfoil profiles on one axes, normalised to unit chord."""
    fig, ax = plt.subplots(figsize=(8, 2.7))

    for desig, coords in shapes.items():
        ax.plot(
            coords[:, 0], coords[:, 1], label=desig, linewidth=0.8,
            marker="o", markersize=1.5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True, linewidth=0.3)

    # Legend below the axes, not on top of the foil shapes
    ax.legend(
        fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.2),
    )
    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")
