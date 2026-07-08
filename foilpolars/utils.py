"""Small helpers not specific to any single pipeline stage."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerosandbox import _asb_root


def save_or_show(
    fig: plt.Figure, save_path: str | None = None, **savefig_kwargs: object,
) -> None:
    """Save a figure to file if given a path, otherwise show it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, **savefig_kwargs)
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


def get_airfoil_database_path() -> Path:
    """Return path to the AeroSandbox airfoil database directory."""
    return _asb_root / "geometry" / "airfoil" / "airfoil_database"


def list_airfoil_names() -> list[str]:
    """Return the sorted names of every airfoil in the AeroSandbox database."""
    db_path = get_airfoil_database_path()
    return sorted(p.stem for p in db_path.glob("*.dat"))


def print_airfoil_database() -> None:
    """Print, then save, the full AeroSandbox airfoil database listing."""
    names = list_airfoil_names()

    # Print summary and full list
    print(f"NeuralFoil training airfoil database: {len(names)} airfoils")
    print(f"Database path: {get_airfoil_database_path()}")
    print("-" * 60)
    for name in names:
        print(name)

    # Save list to the output data directory
    out_path = (
        Path(__file__).parent.parent / "output" / "data" / "nf_airfoils.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n")
    print(f"\nSaved to {out_path}")
