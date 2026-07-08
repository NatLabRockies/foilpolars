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


def _load_airfoil(desig: str) -> "asb.Airfoil":
    """Build an aerosandbox Airfoil, preferring a local coordinate fix."""
    import aerosandbox as asb

    # Some UIUC entries (e.g. naca633218) are known to be wrong; a
    # corrected coordinate file overrides the bundled database if present
    fix_path = os.path.join(FIXES_DIR, f"{desig}.dat")
    if os.path.exists(fix_path):
        coordinates = np.loadtxt(fix_path)
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
    out_dir: str = "output/data/shapes",
    header: str = "x/c,y/c",
    extra_columns: dict[str, dict[str, object]] | None = None,
) -> None:
    """Write each foil's 2-column coordinates to its own csv, plus an index."""
    os.makedirs(out_dir, exist_ok=True)
    foil_ids = list(shapes.keys())

    # One coordinate file per foil, named after its id
    for foil_id in foil_ids:
        path = os.path.join(out_dir, f"{foil_id}.csv")
        np.savetxt(
            path, shapes[foil_id], fmt="%.6f", delimiter=",",
            header=header, comments="",
        )

    # Index file mapping each foil id to its coordinate file, plus any
    # caller-supplied per-foil metadata (e.g. affine transform, PGA
    # coords) merged in as extra columns
    extra_names = list(next(iter(extra_columns.values())).keys()) \
        if extra_columns else []
    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w") as f:
        f.write(",".join(["foil_id", "n_points", "file", *extra_names]))
        f.write("\n")
        for foil_id in foil_ids:
            n_points = shapes[foil_id].shape[0]
            row = [foil_id, n_points, f"{foil_id}.csv"]
            if extra_columns is not None:
                row += [extra_columns[foil_id][name] for name in extra_names]
            f.write(",".join(str(v) for v in row))
            f.write("\n")
    print(f"Saved {len(foil_ids)} shape files to {out_dir}")
    print(f"Saved {index_path}")


def plot_shapes(
    shapes: dict[str, np.ndarray], save_path: str | None = None,
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
