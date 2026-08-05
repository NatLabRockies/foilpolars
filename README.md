# foilpolars

Multifidelity aerodynamic polar data generation for hydrofoil/tidal-turbine
airfoil sections. `foilpolars` ties together three pieces:

- **[AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)** supplies the
  baseline airfoil coordinates (UIUC database).
- **[G2Aero](https://github.com/NatLabRockies/G2Aero)** parameterizes those
  shapes on a Grassmannian manifold (Karcher mean + PGA basis) and samples
  new perturbed shapes around that basis.
- **XFoil** (panel method) and **NeuralFoil** (neural-network surrogate,
  shipped with AeroSandbox) each solve the resulting shapes for lift,
  drag, moment, and pressure at the swept angles of attack, Reynolds
  numbers, and `n_crit` values.

## What it does

The baseline airfoils in the sweep config are loaded via AeroSandbox
(optionally repanelled), then mapped onto a Grassmannian shape space with
G2Aero to compute a Karcher mean and PGA basis. New shapes are sampled
around that basis, and for each perturbed shape, angle of attack, Reynolds
number, and `n_crit` in the config, both solvers are run and the results are
assembled into a single `xarray` dataset (`Cl`, `Cd`, `Cm`, `Cp_min`,
convergence/confidence flags, PGA shape parameters), from which a
cavitation-inception proxy is derived and comparison plots are written
per shape/Re/`n_crit`.

## Layout

- `foilpolars/shapes.py` — loads airfoil coordinates via AeroSandbox,
  applying local coordinate fixes from `foilpolars/airfoil_fixes/`
  where present.
- `foilpolars/grassmann.py` — Grassmannian shape parameterization
  (Karcher mean, PGA basis) and perturbed-shape sampling, via G2Aero.
- `foilpolars/solvers/xfoil_solver.py` — drives the compiled
  `bin/xfoil` binary through a batch script and parses its polar output.
- `foilpolars/solvers/neuralfoil_solver.py` — wraps the NeuralFoil
  surrogate model.
- `foilpolars/sweep.py` — orchestrates the sweep over
  (shape, alpha, Re, n_crit, fidelity) and builds the combined
  `xarray.Dataset`.
- `foilpolars/postprocess.py` — cavitation proxy, convergence
  summary, and plotting (lift/drag polars, Cp_min, confidence maps,
  cavitation bucket).
- `foilpolars/utils.py` — lists airfoils in the AeroSandbox/
  NeuralFoil training database, plus other small shared helpers.
- `foilpolars/cli.py` — `foilpolars` command-line entry point
  (`run`, `get-foils`, `grassmann`, `plot`, `list-foils`).
- `configs/sweep_config.yaml` / `configs/sweep_config_small.yaml` —
  airfoils, repanelling, Grassmann perturbation, alpha/Re/n_crit ranges,
  solver settings, and operating conditions (chord, depth, temperature)
  for a sweep.

## Repanelling

By default each airfoil is repanelled to `n_points_per_side` points per
side (~2x total, default 100/side = 199 points) before being passed to
the solvers. Set
`repanel.enabled: false` in the config to use the raw UIUC coordinates
unchanged instead — both XFoil and NeuralFoil accept arbitrary coordinate
arrays, so this is safe, but irregular point spacing in the raw tables
may affect solver accuracy/convergence.

## Installation

```bash
uv sync
```

This installs AeroSandbox, NeuralFoil, and G2Aero automatically. XFoil is
not on PyPI and must be built separately — see below.

### Installing XFoil

`foilpolars` expects a compiled XFoil binary at `bin/xfoil`. XFoil is
distributed as Fortran source by MIT; see the
[official XFoil page](https://web.mit.edu/drela/Public/web/xfoil/) for
background and the source download. A copy of the source
(`xfoil6.99.tgz`) plus a `gfortran` Makefile are already included in
`xfoil_src/`, so on a Linux box with `gfortran` you can build it with:

```bash
cd xfoil_src/Xfoil/bin
make -f Makefile_gfortran xfoil
cp xfoil ../../../bin/xfoil
```

If you'd rather start from a fresh download, grab the source tarball
from the [XFoil page](https://web.mit.edu/drela/Public/web/xfoil/) and
build it the same way. NeuralFoil-only sweeps don't need XFoil at all —
just set the solver accordingly in the sweep config.

## Usage

```bash
foilpolars run --config configs/sweep_config.yaml
foilpolars get-foils --config configs/sweep_config.yaml
foilpolars grassmann --config configs/sweep_config.yaml
foilpolars plot --config configs/sweep_config.yaml
foilpolars plot-worst-foil --config configs/sweep_config.yaml -n 5
foilpolars list-foils
```

`run` also rebuilds the Grassmann/perturbed-shape artifacts before
sweeping; `grassmann` rebuilds just those artifacts on their own, and
`plot` regenerates every figure except the per-(foil, Re, n_crit)
comparison plots. `plot-worst-foil` reloads a saved sweep dataset and
plots those comparison figures for the `-n` worst-converging foils.
Output dataset and figures are written under `output/`. `submit_full_sweep.sh`
runs the full `grassmann` → `run` → `plot` sequence as a Slurm job.

## License

BSD-3-Clause, see [LICENSE](LICENSE).

## Authors

- Rimple Sandhu
- Andrew Glaws
- Malik Hassanaly
- Bumseok Lee
- Ryan King
