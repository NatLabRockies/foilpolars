"""Command-line interface for foilpolars, installed as `foilpolars`."""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from foilpolars.grassmann import (
    add_pga_columns,
    add_shared_basis_params,
    check_reconstruction,
    compute_grassmann,
    compute_pga_basis,
    load_grassmann_cache,
    perturb_grassmann,
    plot_grassmann_baseline_samples,
    plot_max_thickness_histogram,
    plot_perturbed_shapes,
    plot_pga_pairs,
    plot_te_thickness_histogram,
    save_grassmann_baseline,
    save_grassmann_cache,
    shapes_dict,
)
from foilpolars.postprocess import (
    compute_cavitation_sigma,
    drop_low_quality_foils,
    mask_untrusted_points,
    plot_convergence,
    plot_foil_re_comparison,
    plot_foil_shape,
    plot_pga_pairs_worst,
)
from foilpolars.shapes import (
    load_all_shapes,
    load_raw_shapes,
    plot_shapes,
    save_shapes,
)
from foilpolars.sweep import run_full_sweep

DEFAULT_N_WORST_FOILS = 5


def out_path_for(config_path: str) -> str:
    """Sweep netcdf path derived from the config filename's tag suffix."""
    stem = os.path.splitext(os.path.basename(config_path))[0]
    tag = stem.removeprefix("config")
    return f"output/data{tag}/foilpolars.nc"


def figures_dir_for(out_path: str) -> str:
    """Figures dir mirroring the data dir, e.g. data_small -> figures_small."""
    data_dir = os.path.dirname(out_path)
    return data_dir.replace("data", "figures", 1)


def save_baseline_shapes(
    config: dict,
    data_dir: str,
    figures_dir: str,
    plot: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load, repanel and save the raw + repaneled baseline foils."""
    print("[save_baseline_shapes] loading raw baseline foils")
    raw_shapes = load_raw_shapes(config)
    save_shapes(raw_shapes, out_path=f"{data_dir}/foil_baseline.nc")

    repaneled_shapes = load_all_shapes(config)
    save_shapes(
        repaneled_shapes, out_path=f"{data_dir}/foil_baseline_repanel.nc",
    )
    if plot:
        plot_shapes(
            repaneled_shapes, save_path=f"{figures_dir}/foil_baseline.png",
        )
    return raw_shapes, repaneled_shapes


def baseline(config_path: str) -> None:
    """Load, repanel, save and plot the foils listed in the config."""
    print("[baseline] starting")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = out_path_for(config_path)
    data_dir = os.path.dirname(out_path)
    figures_dir = figures_dir_for(out_path)
    save_baseline_shapes(config, data_dir, figures_dir)


GRASSMANN_OWNED_FILES = [
    "foil_baseline_grass.nc", "foil_perturbed.nc", "foil_perturbed_grass.nc",
]


def grassmann_cache_path(data_dir: str) -> str:
    """Path to the cached shapes/basis/perturbed samples used for replotting."""
    return f"{data_dir}/grassmann_cache.npz"


def build_grassmann_artifacts(
    config: dict,
    data_dir: str,
    figures_dir: str,
) -> dict[str, object]:
    """Rebuild the baseline/Grassmann/perturbed shape artifacts and cache."""
    print("[build_grassmann_artifacts] starting")
    # Delete-then-regenerate: the config's fixed seed always reproduces
    # the same files, so no stale file lingers and reruns match exactly
    for name in GRASSMANN_OWNED_FILES:
        try:
            os.remove(f"{data_dir}/{name}")
        except FileNotFoundError:
            pass

    raw_shapes, repaneled_shapes = save_baseline_shapes(
        config, data_dir, figures_dir, plot=False,
    )

    # Grassmann embedding of the raw baseline shapes
    print("[build_grassmann_artifacts] computing Grassmann embedding")
    grassmann_results = compute_grassmann(raw_shapes)
    check_reconstruction(raw_shapes, grassmann_results)
    save_grassmann_baseline(
        grassmann_results, out_path=f"{data_dir}/foil_baseline_grass.nc",
    )

    # PGA needs equal landmark counts, so use repaneled shapes with
    # r=4 modes, matching Doronina et al. 2022's eigenvalue-decay cutoff
    print("[build_grassmann_artifacts] computing PGA basis")
    basis = compute_pga_basis(repaneled_shapes, n_coord=4)
    grassmann_config = config.get("grassmann", {})
    print("[build_grassmann_artifacts] sampling perturbed shapes")
    perturbed = perturb_grassmann(
        basis,
        n_perturb=grassmann_config.get("n_perturb", 20),
        seed=grassmann_config.get("seed"),
    )

    # Physical and Grassmann-space coords of the same perturbed samples,
    # saved separately so neither file is ambiguous about its space
    phys_shapes = shapes_dict(perturbed["phys"])
    save_shapes(phys_shapes, out_path=f"{data_dir}/foil_perturbed.nc")
    grass_shapes = shapes_dict(perturbed["X_gr"])
    save_shapes(
        grass_shapes, out_path=f"{data_dir}/foil_perturbed_grass.nc",
        columns=("Xgr_0", "Xgr_1"),
    )

    # Cache everything `plot` needs to redraw the Grassmann figures
    # later without repeating the PGA/Karcher computation above
    save_grassmann_cache(
        grassmann_cache_path(data_dir), repaneled_shapes, basis, perturbed,
    )

    return {
        "basis": basis, "perturbed": perturbed,
        "foil_perturbed": phys_shapes,
    }


def grassmann(config_path: str) -> None:
    """Rebuild the baseline/Grassmann/perturbed shape artifacts."""
    print("[grassmann] starting")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = out_path_for(config_path)
    data_dir = os.path.dirname(out_path)
    figures_dir = figures_dir_for(out_path)
    build_grassmann_artifacts(config, data_dir, figures_dir)
    print("Run `foilpolars plot` to (re)generate figures from this data.")


def plot_comparison_figures(
    ds: xr.Dataset,
    config: dict,
    figures_dir: str,
    n_foils: int = DEFAULT_N_WORST_FOILS,
) -> None:
    """Shape + per-(Re, n_crit) comparison figures for the n_foils worst."""
    print("[plot_comparison_figures] starting")
    foil_ids_all = ds["foil_id"].values
    xfoil_frac = ds["converged"].sel(fidelity="xfoil").mean(
        dim=("alpha", "Re", "n_crit")
    ).values
    n = min(n_foils, len(foil_ids_all))
    worst = np.argsort(xfoil_frac)[:n]
    foil_ids = foil_ids_all[worst]
    if len(foil_ids_all) > n_foils:
        print(
            f"{len(foil_ids_all)} perturbed shapes exceeds the "
            f"{n_foils}-shape cap: plotting only the "
            f"{n_foils} worst-converging"
        )

    # One shape outline per worst-converging foil, in its own directory
    for foil_id in foil_ids:
        plot_foil_shape(ds, foil_id, figures_dir=figures_dir)

    # Re/n_crit come from the dataset itself, not the live config's
    # `sweep` block — those are fixed at sweep time and must never
    # drift from what's actually in the saved data
    cav = config["cavitation"]
    re_values = [float(r) for r in ds["Re"].values]
    chord = float(cav["chord"])
    depth = float(cav["depth"])
    temperature = float(cav["temperature"])
    sigma_by_re = compute_cavitation_sigma(
        re_values, chord, depth, temperature,
    )
    combos = list(itertools.product(foil_ids, re_values, ds["n_crit"].values))
    for foil_id, re, n_crit in tqdm(
        combos, desc="[plot_comparison_figures] figures",
    ):
        plot_foil_re_comparison(
            ds, foil_id, re, sigma_by_re[re],
            n_crit=float(n_crit), figures_dir=figures_dir,
        )


def plot_grassmann_figures(
    data_dir: str,
    figures_dir: str,
    min_te_thickness: float = 1e-4,
    min_thickness: float = 0.05,
) -> None:
    """Redraw every Grassmann figure from the cached shapes/basis/samples."""
    print("[plot_grassmann_figures] starting")
    repaneled_shapes, basis, perturbed = load_grassmann_cache(
        grassmann_cache_path(data_dir),
    )
    plot_shapes(
        repaneled_shapes, save_path=f"{figures_dir}/foil_baseline.png",
    )
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
    plot_te_thickness_histogram(
        perturbed, save_path=f"{figures_dir}/te_thickness.png",
        min_te_thickness=min_te_thickness,
    )
    plot_max_thickness_histogram(
        perturbed, save_path=f"{figures_dir}/max_thickness.png",
        min_thickness=min_thickness,
    )


def plot(
    config_path: str,
    n_foils: int | None = None,
) -> None:
    """Regenerate every figure from whatever saved data/cache exists."""
    print("[plot] starting")
    out_path = out_path_for(config_path)
    data_dir = os.path.dirname(out_path)
    figures_dir = figures_dir_for(out_path)

    # Only `cavitation`/`postprocess` config values are used below —
    # `sweep` params (Re, n_crit, ...) always come from the dataset
    # itself, since they're fixed once the sweep has run
    with open(config_path) as f:
        config = yaml.safe_load(f)
    pp = config.get("postprocess", {})
    if n_foils is None:
        n_foils = pp.get("n_worst_foils_for_plot", DEFAULT_N_WORST_FOILS)
    min_xfoil_conv = float(pp.get("min_xfoil_conv", 0.75))
    min_nf_conf = float(pp.get("min_neuralfoil_confidence", 0.75))
    min_te_thickness = float(pp.get("min_te_thickness", 1e-4))
    min_thickness = float(pp.get("min_thickness", 0.05))

    # Grassmann figures need `grassmann`'s cache; sweep figures need
    # `sweep`'s saved dataset, reloaded here even when `sweep` calls
    # this directly, so every figure reflects what actually landed on
    # disk
    cache_path = grassmann_cache_path(data_dir)
    if os.path.exists(cache_path):
        plot_grassmann_figures(
            data_dir, figures_dir, min_te_thickness, min_thickness,
        )

    if os.path.exists(out_path):
        ds = xr.open_dataset(out_path)
        plot_convergence(
            ds, fname=f"{figures_dir}/convergence.png",
            min_xfoil_conv=min_xfoil_conv,
            min_neuralfoil_confidence=min_nf_conf,
        )
        plot_pga_pairs_worst(
            ds, fname=f"{figures_dir}/pga_pairs_worst.png",
            min_xfoil_conv=min_xfoil_conv,
        )
        plot_comparison_figures(ds, config, figures_dir, n_foils)


def sweep(config_path: str) -> None:
    """Rebuild shape artifacts, then sweep XFoil + NeuralFoil over them."""
    print("[sweep] starting")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_path = out_path_for(config_path)
    data_dir = os.path.dirname(out_path)
    figures_dir = figures_dir_for(out_path)
    os.makedirs(data_dir, exist_ok=True)

    # Same pipeline as `grassmann` — the sweep runs over its perturbed
    # shapes, not the baseline foils directly
    print("[sweep] building Grassmann/perturbed-shape artifacts")
    artifacts = build_grassmann_artifacts(config, data_dir, figures_dir)
    shapes = artifacts["foil_perturbed"]

    # Both solvers over every (shape, alpha, Re) combination, checkpointed
    # to out_path after each foil so a killed job keeps completed ones
    print("[sweep] running the XFoil + NeuralFoil sweep")
    ds = run_full_sweep(config, shapes, checkpoint_path=out_path)

    # Attach each shape's 5 PGA params, then the shared Karcher
    # mean/PGA basis/affine transform needed to reconstruct any shape
    ds = add_pga_columns(ds, artifacts["perturbed"])
    ds = add_shared_basis_params(ds, artifacts["basis"])
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")

    # Every figure is redrawn from what was just saved to disk, same as
    # a standalone `plot` call, capped at postprocess.n_worst_foils or
    # the comparison figures would number in the tens of thousands
    plot(config_path)


def save(config_path: str) -> None:
    """Drop low-quality foils, then mask points untrusted by either solver."""
    print("[save] starting")
    with open(config_path) as f:
        pp = yaml.safe_load(f).get("postprocess", {})
    min_xfoil_conv = float(pp.get("min_xfoil_conv", 0.75))
    min_te_thickness = float(pp.get("min_te_thickness", 1e-4))
    min_thickness = float(pp.get("min_thickness", 0.05))
    min_nf_conf = float(pp.get("min_neuralfoil_confidence", 0.75))

    out_path = out_path_for(config_path)
    ds = xr.open_dataset(out_path)

    # Foil-level first: drops whole foils before the (larger) point-level
    # mask runs over them, so no wasted work on foils about to be dropped
    ds = drop_low_quality_foils(
        ds, min_xfoil_conv, min_te_thickness, min_thickness,
    )
    ds = mask_untrusted_points(ds, min_nf_conf)

    stem, ext = os.path.splitext(out_path)
    clean_path = f"{stem}_clean{ext}"
    ds.to_netcdf(clean_path)
    print(f"Saved {clean_path}")


def main() -> None:
    """Entry point for the `foilpolars` command."""
    parser = argparse.ArgumentParser(prog="foilpolars")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_parser = subparsers.add_parser(
        "sweep", help="Run the XFoil + NeuralFoil sweep and plot figures"
    )
    sweep_parser.add_argument(
        "-c", "--config", default="configs/config.yaml"
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Load, repanel, save and plot the configured foils",
    )
    baseline_parser.add_argument(
        "-c", "--config", default="configs/config.yaml"
    )

    grassmann_parser = subparsers.add_parser(
        "grassmann",
        help="Convert raw foil shapes to a Grassmann parameterization",
    )
    grassmann_parser.add_argument(
        "-c", "--config", default="configs/config.yaml"
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Regenerate every figure from whatever saved data exists",
    )
    plot_parser.add_argument(
        "-c", "--config", default="configs/config.yaml"
    )
    plot_parser.add_argument(
        "-n", "--n-foils", type=int, default=None,
        help="default: postprocess.n_worst_foils from the config",
    )

    save_parser = subparsers.add_parser(
        "save",
        help="Drop low-quality foils/points, save a cleaned dataset",
    )
    save_parser.add_argument(
        "-c", "--config", default="configs/config.yaml"
    )

    args = parser.parse_args()
    if args.command == "sweep":
        sweep(args.config)
    elif args.command == "baseline":
        baseline(args.config)
    elif args.command == "grassmann":
        grassmann(args.config)
    elif args.command == "plot":
        plot(args.config, args.n_foils)
    elif args.command == "save":
        save(args.config)


if __name__ == "__main__":
    main()
