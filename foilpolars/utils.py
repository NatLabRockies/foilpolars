"""Small helpers not specific to any single pipeline stage."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_or_show(
    fig: plt.Figure,
    save_path: str | None = None,
    quiet: bool = False,
    **savefig_kwargs: object,
) -> None:
    """Save a figure to file if given a path, otherwise show it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, **savefig_kwargs)
        plt.close(fig)
        if not quiet:
            print(f"Saved {save_path}")
    else:
        plt.show()
