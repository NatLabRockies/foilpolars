"""Command-line interface for foilpolars, installed as `foilpolars`."""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import xarray as xr
import yaml

from foilpolars.grassmann import (
    add_pga_columns,
    add_shared_basis_params,
    check_reconstruction,
    compute_grassmann,
    compute_pga_basis,
    perturb_grassmann,
    plot_grassmann_baseline_samples,
    plot_perturbed_shapes,
    plot_pga_pairs,
    save_grassmann_baseline,
    shapes_dict,
)
from foilpolars.postprocess import (
    compute_cavitation_sigma,
    plot_foil_re_comparison,
    save_full_results,
    summarize_convergence,
)
from foilpolars.shapes import (
    load_all_shapes,
    load_raw_shapes,
    plot_shapes,
    save_shapes,
)
from foilpolars.sweep import run_full_sweep
from foilpolars.utils import print_airfoil_database

MAX_COMPARISON_PLOT_SHAPES = 20


def figures_dir_for(config: dict) -> str:
    """Figures dir mirroring the data dir, e.g. data_small -> figures_small."""
    data_dir = os.path.dirname(config["output"]["path"])
    return data_dir.replace("data", "figures", 1)


def save_baseline_shapes(
    config: dict, data_dir: str, figures_dir: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load, repanel, save and plot the raw + repaneled baseline foils."""
    raw_shapes = load_raw_shapes(config)
    save_shapes(raw_shapes, out_dir=f"{data_dir}/foil_baseline")

    repaneled_shapes = load_all_shapes(config)
    save_shapes(
        repaneled_shapes, out_dir=f"{data_dir}/foil_baseline_repanel",
    )
    plot_shapes(
        repaneled_shapes, save_path=f"{figures_dir}/foil_baseline.png",
    )
    return raw_shapes, repaneled_shapes


def get_foils(config_path: str) -> None:
    """Load, repanel, save and plot the foils listed in the config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = os.path.dirname(config["output"]["path"])
    figures_dir = figures_dir_for(config)
    save_baseline_shapes(config, data_dir, figures_dir)


GRASSMANN_OWNED_DIRS = [
    "foil_baseline_grass", "foil_perturbed", "foil_perturbed_grass",
]


def build_grassmann_artifacts(
    config: dict, data_dir: str, figures_dir: str,
) -> dict[str, object]:
    """Rebuild the baseline/Grassmann/perturbed shape artifacts and plots."""
    # Delete-then-regenerate: with the config's fixed seed this always
    # reproduces the same files, so stale foils from a previous config
    # never linger and reruns are exactly reproducible
    for name in GRASSMANN_OWNED_DIRS:
        shutil.rmtree(f"{data_dir}/{name}", ignore_errors=True)

    raw_shapes, repaneled_shapes = save_baseline_shapes(
        config, data_dir, figures_dir,
    )

    # Grassmann embedding of the raw baseline shapes
    grassmann_results = compute_grassmann(raw_shapes)
    check_reconstruction(raw_shapes, grassmann_results)
    save_grassmann_baseline(
        grassmann_results, out_dir=f"{data_dir}/foil_baseline_grass",
    )

    # PGA needs equal landmark counts, so use repaneled shapes. Use
    # r=4 principal geodesic modes, matching the eigenvalue-decay
    # truncation used in the Grassmannian airfoil paper (Doronina
    # et al. 2022) for perturbing shapes
    basis = compute_pga_basis(repaneled_shapes, n_coord=4)
    grassmann_config = config.get("grassmann", {})
    perturbed = perturb_grassmann(
        basis,
        n_perturb=grassmann_config.get("n_perturb", 20),
        seed=grassmann_config.get("seed"),
    )

    # Physical and Grassmann-space coordinates of the same perturbed
    # samples, saved to separate directories so neither format is
    # ambiguous about which space its coordinates live in
    phys_shapes = shapes_dict(perturbed["phys"])
    save_shapes(phys_shapes, out_dir=f"{data_dir}/foil_perturbed")
    grass_shapes = shapes_dict(perturbed["X_gr"])
    save_shapes(
        grass_shapes, out_dir=f"{data_dir}/foil_perturbed_grass",
        header="Xgr_0,Xgr_1",
    )

    # Plot baseline vs. perturbed shapes in physical x/y coordinates
    # and on the Grassmann manifold
    plot_perturbed_shapes(
        repaneled_shapes, basis, perturbed,
        save_path=f"{figures_dir}/foil_perturbed.png",
    )
    plot_grassmann_baseline_samples(
        basis, perturbed,
        save_path=f"{figures_dir}/foil_perturbed_grass.png",
    )
    plot_pga_pairs(
        basis, perturbed, save_path=f"{figures_dir}/pga_pairs.png",
    )

    return {
        "basis": basis, "perturbed": perturbed,
        "foil_perturbed": phys_shapes,
    }


def grassmann(config_path: str) -> None:
    """Rebuild the baseline/Grassmann/perturbed shape artifacts and plots."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = os.path.dirname(config["output"]["path"])
    figures_dir = figures_dir_for(config)
    build_grassmann_artifacts(config, data_dir, figures_dir)


def plot_comparison_figures(
    ds: xr.Dataset, config: dict, figures_dir: str,
) -> None:
    """One comparison figure per (foil, Re, n_crit), skipped if too many."""
    if len(ds["foil_id"]) > MAX_COMPARISON_PLOT_SHAPES:
        print(
            f"Skipping per-shape comparison plots: "
            f"{len(ds['foil_id'])} perturbed shapes exceeds the "
            f"{MAX_COMPARISON_PLOT_SHAPES}-shape cap"
        )
        return

    op = config["operating"]
    cav = config["cavitation"]
    re_values = [float(r) for r in op["reynolds"]]
    chord = float(cav["chord"])
    depth = float(cav["depth"])
    temperature = float(cav["temperature"])
    sigma_by_re = compute_cavitation_sigma(
        re_values, chord, depth, temperature,
    )
    for foil_id in ds["foil_id"].values:
        for re in re_values:
            for n_crit in ds["n_crit"].values:
                plot_foil_re_comparison(
                    ds, foil_id, re, sigma_by_re[re],
                    n_crit=float(n_crit), figures_dir=figures_dir,
                )


def plot(config_path: str) -> None:
    """Reload a saved sweep dataset and regenerate its figures."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = config["output"]["path"]
    ds = xr.open_dataset(out_path)
    plot_comparison_figures(ds, config, figures_dir_for(config))


def run(config_path: str) -> None:
    """Rebuild shape artifacts, then sweep XFoil + NeuralFoil over them."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = config["output"]["path"]
    data_dir = os.path.dirname(out_path)
    figures_dir = figures_dir_for(config)
    os.makedirs(data_dir, exist_ok=True)

    # Same baseline/Grassmann/perturbed-shape pipeline as the `grassmann`
    # command — the sweep runs over its perturbed shapes, not the
    # baseline foils directly
    artifacts = build_grassmann_artifacts(config, data_dir, figures_dir)
    shapes = artifacts["foil_perturbed"]

    # Run both solvers over every (perturbed shape, alpha, Re) combination,
    # checkpointing the raw results to the output nc path after each foil
    # finishes so a killed job doesn't lose completed foils
    ds = run_full_sweep(config, shapes, checkpoint_path=out_path)

    # Print the convergence/confidence summary (the full csv below already
    # carries per-row convergence/confidence, so no separate file for it),
    # then attach each shape's 5 varying PGA params and dump the full
    # results table. Only the varying params go into the csv, since the
    # shared basis added below would blow up its per-row flattening
    print(summarize_convergence(ds))
    ds = add_pga_columns(ds, artifacts["perturbed"])
    save_full_results(ds, f"{data_dir}/foilpolars.csv")

    # Attach the shared Karcher mean / PGA basis / mean affine transform
    # needed to reconstruct each shape, then persist the combined dataset
    # of polars + shape-reconstruction params
    ds = add_shared_basis_params(ds, artifacts["basis"])
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")

    # One comparison figure per (foil, Re, n_crit), at the fixed
    # cavitation chord/depth/temperature. Skipped when n_perturb is large,
    # since one figure per perturbed shape per Re/n_crit would otherwise
    # number in the tens of thousands
    plot_comparison_figures(ds, config, figures_dir)


def main() -> None:
    """Entry point for the `foilpolars` command."""
    parser = argparse.ArgumentParser(prog="foilpolars")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the XFoil + NeuralFoil sweep and plot figures"
    )
    run_parser.add_argument("--config", default="configs/sweep_config.yaml")

    subparsers.add_parser(
        "list-foils", help="List the AeroSandbox airfoil database"
    )

    get_foils_parser = subparsers.add_parser(
        "get-foils",
        help="Load, repanel, save and plot the configured foils",
    )
    get_foils_parser.add_argument(
        "--config", default="configs/sweep_config.yaml"
    )

    grassmann_parser = subparsers.add_parser(
        "grassmann",
        help="Convert raw foil shapes to a Grassmann parameterization",
    )
    grassmann_parser.add_argument(
        "--config", default="configs/sweep_config.yaml"
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Regenerate comparison figures from a saved sweep dataset",
    )
    plot_parser.add_argument("--config", default="configs/sweep_config.yaml")

    args = parser.parse_args()
    if args.command == "run":
        run(args.config)
    elif args.command == "list-foils":
        print_airfoil_database()
    elif args.command == "get-foils":
        get_foils(args.config)
    elif args.command == "grassmann":
        grassmann(args.config)
    elif args.command == "plot":
        plot(args.config)


if __name__ == "__main__":
    main()
