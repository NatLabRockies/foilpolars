# FOILPOLARS: Grassmannian Foil Shape Sweeps for Polar Generation

NLR Software Record: SWR 26-095

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

## Repanelling

Each airfoil is repanelled to `grassmann.n_repanel_points_per_side`
points per side (~2x total, default 100/side = 199 points) before being
mapped onto the Grassmannian; PGA needs equal landmark counts across
foils, so repanelling can't be disabled.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment and dependencies. If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(see the [uv install docs][uv-install] for other platforms/methods,
e.g. `pipx install uv` or `brew install uv`). Then, from the repo root:

```bash
uv sync
```

This installs AeroSandbox, NeuralFoil, and G2Aero automatically. XFoil is
not on PyPI and must be built separately — see below.

`uv sync` creates the environment at `.venv/` but doesn't activate it.
Either activate it once per shell:

```bash
source .venv/bin/activate
foilpolars sweep --config configs/config.yaml
```

or prefix each command with `uv run` (no activation needed, `run.sh`
below uses this form):

```bash
uv run foilpolars sweep --config configs/config.yaml
```

### Installing XFoil

Only needed if the XFoil solver is enabled in your sweep config; skip
this section for NeuralFoil-only sweeps. When enabled, `foilpolars`
looks for a compiled XFoil binary at `bin/xfoil`. XFoil is distributed
as Fortran source by MIT and is not vendored in this repo (`xfoil_src/`
and `bin/xfoil` are gitignored) — download and build it yourself:

```bash
mkdir -p xfoil_src && cd xfoil_src
curl -O https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz
tar xzf xfoil6.99.tgz
cd Xfoil/bin
make -f Makefile_gfortran xfoil
cp xfoil ../../../bin/xfoil
```

This needs `gfortran` on a Linux box; see the
[official XFoil page](https://web.mit.edu/drela/Public/web/xfoil/) for
background if the build fails.

## Usage

All commands take `--config configs/config.yaml` (or another
config path); output dataset and figures are written under `output/`.

```bash
foilpolars baseline --config configs/config.yaml
foilpolars grassmann --config configs/config.yaml
foilpolars sweep --config configs/config.yaml
foilpolars plot --config configs/config.yaml -n 5
foilpolars save --config configs/config.yaml
```

- `baseline` — loads, repanels, saves, and plots the `baseline` foils
  named in the config.
- `grassmann` — maps the baseline foils onto a Grassmannian shape
  space (Karcher mean + PGA basis) and samples the perturbed shapes,
  caching the result.
- `sweep` — rebuilds the Grassmann/perturbed-shape artifacts, runs the
  full XFoil + NeuralFoil sweep over shape/alpha/Re/`n_crit`, saves
  the dataset, then calls `plot`.
- `plot` — regenerates every figure from whatever saved data/cache
  exists, including per-(foil, Re, n_crit) comparisons for the `-n`
  worst-converging foils.
- `save` — drops whole foils below `postprocess.min_xfoil_conv`,
  `min_te_thickness`, or `min_thickness`, then marks every remaining
  (foil, alpha, Re, n_crit) point untrusted — via a shared `converged`
  flag on both fidelities — unless XFoil converged there *and*
  NeuralFoil's confidence meets `min_neuralfoil_confidence`; writes
  `{stem}_clean.nc` beside the sweep dataset.

## Running the full pipeline

`run.sh` chains `sweep` then `save` for one config, selected by an
optional tag (no tag = `configs/config.yaml`):

```bash
./run.sh          # configs/config.yaml
./run.sh small    # configs/config_small.yaml
./run.sh mhk      # configs/config_mhk.yaml
```

## License

BSD-3-Clause, see [LICENSE](LICENSE).

## Disclaimer

THIS SOFTWARE WAS GENERATED USING ARTIFICIAL INTELLIGENCE. IT MAY
CONTAIN ERRORS OR INACCURACIES. THIS SOFTWARE IS PROVIDED "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE ALLIANCE FOR ENERGY
INNOVATION, LLC OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE. USE AT YOUR OWN RISK.

## Authors

- Rimple Sandhu, Computational Science Center, National Laboratory of the Rockies (NLR)
- Andrew Glaws, Computational Science Center, National Laboratory of the Rockies (NLR)
- Malik Hassanaly, Computational Science Center, National Laboratory of the Rockies (NLR)
- Bumseok Lee, National Wind Technology Center, National Laboratory of the Rockies (NLR)
- Ryan King, Computational Science Center, National Laboratory of the Rockies (NLR)

[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
