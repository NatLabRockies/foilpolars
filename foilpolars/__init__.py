"""Multifidelity hydrofoil aerodynamic data generation."""

from foilpolars.postprocess import (
    compute_cavitation_sigma,
    plot_convergence,
    plot_foil_re_comparison,
    summarize_convergence,
)
from foilpolars.shapes import load_all_shapes
from foilpolars.sweep import run_full_sweep

__all__ = [
    "compute_cavitation_sigma",
    "plot_convergence",
    "plot_foil_re_comparison",
    "summarize_convergence",
    "load_all_shapes",
    "run_full_sweep",
]
