"""Aerodynamic solvers: NeuralFoil and XFoil wrappers."""

from foilpolars.solvers.neuralfoil_solver import run_neuralfoil
from foilpolars.solvers.xfoil_solver import run_xfoil

__all__ = ["run_neuralfoil", "run_xfoil"]
