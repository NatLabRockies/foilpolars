"""Post-processing utilities: cavitation proxy, convergence summary, plots."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foilpolars.utils import save_or_show

_SWEEP_DIMS = ("alpha", "Re", "n_crit")
_SWEEP_LABELS = {
    "alpha": r"$\alpha$ (deg)", "Re": "Re", "n_crit": "$N_{crit}$",
}


def compute_cavitation_sigma(
    re_values: list[float],
    chord: float,
    depth: float,
    temperature: float,
) -> dict[float, float]:
    """Cavitation index sigma per Re, at a fixed chord/depth/temperature."""
    # Water density, viscosity and vapor pressure at temperature t_c,
    # plus the ambient pressure at the given depth
    t_c = temperature
    rho_w = 999.842 - 0.0708 * t_c - 0.00379 * t_c ** 2
    mu_w = 1.787e-3 * np.exp(
        -0.03144 * t_c + 1.69e-4 * t_c ** 2 - 2.02e-6 * t_c ** 3
    )
    pv_w = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    p_inf = 101_325.0 + rho_w * 9.81 * depth

    # Cavitation index at each Reynolds number's implied flow speed
    sigma = {}
    for re in re_values:
        v = re * mu_w / (rho_w * chord)
        sigma[re] = (p_inf - pv_w) / (0.5 * rho_w * v ** 2)
    return sigma


def summarize_convergence(ds: xr.Dataset) -> pd.DataFrame:
    """Summarise XFoil convergence fraction and NeuralFoil mean confidence."""
    print("[summarize_convergence] starting")
    records = []

    for foil_id in ds["foil_id"].values:
        for re in ds["Re"].values:
            for n_crit in ds["n_crit"].values:
                row: dict = {
                    "foil_id": foil_id, "Re": float(re),
                    "n_crit": float(n_crit),
                }

                # XFoil convergence fraction
                if "xfoil" in ds["fidelity"].values:
                    conv = ds["converged"].sel(
                        foil_id=foil_id, Re=re, n_crit=n_crit,
                        fidelity="xfoil",
                    ).values
                    valid = conv[np.isfinite(conv)]
                    row["xfoil_converge_fraction"] = (
                        float(valid.mean()) if len(valid) > 0
                        else float("nan")
                    )
                else:
                    row["xfoil_converge_fraction"] = float("nan")

                # NeuralFoil mean analysis confidence
                if "neuralfoil" in ds["fidelity"].values:
                    conf = ds["analysis_confidence"].sel(
                        foil_id=foil_id, Re=re, n_crit=n_crit,
                        fidelity="neuralfoil",
                    ).values
                    valid_c = conf[np.isfinite(conf)]
                    row["neuralfoil_mean_confidence"] = (
                        float(valid_c.mean()) if len(valid_c) > 0
                        else float("nan")
                    )
                else:
                    row["neuralfoil_mean_confidence"] = float("nan")

                records.append(row)

    df = pd.DataFrame(records).set_index(["foil_id", "Re", "n_crit"])
    return df


def drop_low_quality_foils(
    ds: xr.Dataset,
    min_xfoil_conv: float = 0.75,
    min_te_thickness: float = 1e-4,
    min_thickness: float = 0.05,
) -> xr.Dataset:
    """Drop whole foils with low XFoil convergence or an unbuildable shape."""
    print("[drop_low_quality_foils] starting")
    from foilpolars.grassmann import _te_thickness, reconstruct_phys_shape

    # XFoil convergence fraction per foil, same criterion as
    # plot_convergence's threshold line
    xfoil_conv = ds["converged"].sel(fidelity="xfoil")
    conv_frac = xfoil_conv.mean(dim=("alpha", "Re", "n_crit")).values

    # Reconstruct each foil's physical shape from its saved PGA params to
    # get its trailing-edge gap (not stored directly in the dataset);
    # max thickness is already stored as thickness_max
    mu = ds["mu"].values
    vh = ds["Vh"].values
    m_mean = ds["m_mean"].values
    b_mean = ds["b_mean"].values
    n_coord = sum(
        1 for name in ds.data_vars if str(name).startswith("pga_coef_")
    )
    coef_all = np.column_stack([
        ds[f"pga_coef_{j}"].values for j in range(n_coord)
    ])
    ratio_all = ds["thickness_ratio"].values
    foil_ids = ds["foil_id"].values
    te_thickness = np.array([
        _te_thickness(
            reconstruct_phys_shape(
                mu, vh, m_mean, b_mean, coef_all[i], ratio_all[i],
            )
        )
        for i in range(len(foil_ids))
    ])
    max_thickness = ds["thickness_max"].values

    # Keep only foils meeting all three shape/convergence criteria
    keep = (
        (conv_frac >= min_xfoil_conv)
        & (te_thickness >= min_te_thickness)
        & (max_thickness >= min_thickness)
    )
    n_dropped = int((~keep).sum())
    print(
        f"Dropping {n_dropped}/{len(foil_ids)} foils "
        f"(XFoil converged < {min_xfoil_conv:.2f}, "
        f"TE thickness < {min_te_thickness:.0e}, or "
        f"max thickness < {min_thickness:.2f})"
    )
    return ds.sel(foil_id=foil_ids[keep])


def mask_untrusted_points(
    ds: xr.Dataset,
    min_neuralfoil_confidence: float = 0.75,
) -> xr.Dataset:
    """Zero `converged` wherever XFoil/NeuralFoil isn't jointly trusted."""
    print("[mask_untrusted_points] starting")
    xfoil_ok = ds["converged"].sel(fidelity="xfoil").values == 1.0
    nf_conf = ds["analysis_confidence"].sel(fidelity="neuralfoil").values
    neuralfoil_ok = nf_conf >= min_neuralfoil_confidence

    # Common mask broadcasts back over the fidelity dim: 'converged' is
    # NaN on the neuralfoil slice as saved by the sweep, so this is the
    # one flag both fidelities can share to mark a point as usable
    common = xfoil_ok & neuralfoil_ok
    n_common = int(common.sum())
    print(
        f"{n_common}/{common.size} points trusted by both XFoil "
        f"(converged) and NeuralFoil (confidence >= "
        f"{min_neuralfoil_confidence:.2f})"
    )

    new_converged = np.broadcast_to(
        common[..., None], ds["converged"].shape,
    ).astype(ds["converged"].dtype)
    ds = ds.assign(converged=(ds["converged"].dims, new_converged))
    return ds


def _boxplot_by_foil(
    ax: plt.Axes,
    nf_conf: xr.DataArray,
    foil_ids: np.ndarray,
) -> None:
    """NeuralFoil confidence box plot, one box per foil, pooling sweep dims."""
    samples = [
        nf_conf.sel(foil_id=fid).values.ravel() for fid in foil_ids
    ]
    bplot = ax.boxplot(
        samples, positions=np.arange(len(foil_ids)), widths=0.6,
        patch_artist=True, showfliers=False, whis=0,
        medianprops={"color": "black"},
        whiskerprops={"linewidth": 0}, capprops={"linewidth": 0},
    )
    for patch in bplot["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.6)


def plot_convergence(
    ds: xr.Dataset,
    fname: str = "output/figures/convergence.png",
    n_worst_foils: int = 1000,
    min_xfoil_conv: float = 0.75,
    min_neuralfoil_confidence: float = 0.75,
) -> None:
    """XFoil convergence and NeuralFoil confidence vs sweep params + foil."""
    print("[plot_convergence] starting")
    xfoil_conv = ds["converged"].sel(fidelity="xfoil")
    nf_conf = ds["analysis_confidence"].sel(fidelity="neuralfoil")

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 3)

    # Top row: XFoil convergence + NeuralFoil confidence overlaid,
    # averaged over the other two sweep dims, vs alpha/Re/n_crit
    for i, dim in enumerate(_SWEEP_DIMS):
        ax = fig.add_subplot(gs[0, i])
        other_dims = [d for d in _SWEEP_DIMS if d != dim]
        xfoil_frac = xfoil_conv.mean(dim=("foil_id", *other_dims))
        nf_mean = nf_conf.mean(dim=("foil_id", *other_dims))
        ax.plot(
            ds[dim].values, xfoil_frac.values, "o-", color="black",
            markersize=4, lw=1.5, label="XFoil converged",
        )
        ax.plot(
            ds[dim].values, nf_mean.values, "o-", color="tab:blue",
            markersize=4, lw=1.5, label="NeuralFoil confidence",
        )
        ax.set_xlabel(_SWEEP_LABELS[dim])
        ax.set_ylabel("Convergence fraction / confidence")
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.4)
        ax.legend(fontsize=7)
        if dim == "Re":
            ax.set_xscale("log")

    # Rank foils by XFoil convergence and keep only the worst
    # n_worst_foils, so the bottom two rows stay readable at any scale
    foil_ids_all = ds["foil_id"].values
    xfoil_frac_all = xfoil_conv.mean(dim=("alpha", "Re", "n_crit")).values
    n = min(n_worst_foils, len(foil_ids_all))
    worst = np.argsort(xfoil_frac_all)[:n]
    foil_ids = foil_ids_all[worst]
    xfoil_frac_foil = xfoil_frac_all[worst]
    x = np.arange(n)
    # Thin the tick labels when there are many foils so they don't overlap
    stride = max(1, n // 40)

    # Middle row: XFoil convergence fraction, one point per foil shape,
    # spanning the full figure width
    ax_xfoil = fig.add_subplot(gs[1, :])
    style = "o-" if n <= 40 else "-"
    ax_xfoil.plot(
        x, xfoil_frac_foil, style, color="black", markersize=5,
        lw=1.2, label="XFoil converged",
    )

    # Threshold line, plus a crossing marker (if within the plotted
    # subset) and a count of foils in the full dataset below it
    n_below = int(np.sum(xfoil_frac_all < min_xfoil_conv))
    ax_xfoil.axhline(
        min_xfoil_conv, color="tab:red", linestyle="--", linewidth=1,
        label=f"{min_xfoil_conv:.2f} threshold",
    )
    below = np.nonzero(xfoil_frac_foil < min_xfoil_conv)[0]
    if len(below):
        ax_xfoil.axvline(
            x[below[-1]] + 0.5, color="tab:red", linestyle="--",
            linewidth=1,
        )
    ax_xfoil.text(
        0.01, 0.05,
        f"{n_below}/{len(foil_ids_all)} foils below "
        f"{min_xfoil_conv:.2f}",
        transform=ax_xfoil.transAxes, fontsize=8, color="tab:red",
    )

    ax_xfoil.set_xticks(x[::stride])
    ax_xfoil.set_xticklabels([])
    ax_xfoil.set_xlim(x[0], x[-1])
    ax_xfoil.set_ylim(0, 1)
    ax_xfoil.set_ylabel("XFoil converged (fraction)")
    ax_xfoil.grid(True, linewidth=0.4)
    ax_xfoil.legend(fontsize=8)

    # Bottom row: NeuralFoil confidence, one box per foil shape, pooling
    # alpha/Re/n_crit into each box's sample, with its own threshold line
    ax_nf = fig.add_subplot(gs[2, :])
    _boxplot_by_foil(ax_nf, nf_conf, foil_ids)
    ax_nf.axhline(
        min_neuralfoil_confidence, color="tab:red", linestyle="--",
        linewidth=1, label=f"{min_neuralfoil_confidence:.2f} threshold",
    )
    ax_nf.set_xticks(x[::stride])
    ax_nf.set_xticklabels(
        foil_ids[::stride], rotation=45, ha="right", fontsize=8,
    )
    ax_nf.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax_nf.set_ylim(0, 1)
    ax_nf.set_xlabel(
        f"Foil shape ({n} worst XFoil convergence, ascending)"
    )
    ax_nf.set_ylabel("NeuralFoil confidence")
    ax_nf.grid(True, linewidth=0.4, axis="y")
    ax_nf.legend(fontsize=8)

    fig.tight_layout()
    save_or_show(fig, fname, dpi=150, bbox_inches="tight")


def plot_foil_shape(
    ds: xr.Dataset,
    foil_id: str,
    figures_dir: str = "output/figures",
) -> None:
    """Plot one foil's physical (x/c, y/c) outline, saved in its own dir."""
    from foilpolars.grassmann import reconstruct_phys_shape

    # Shared PGA basis/affine params needed to reconstruct this foil's
    # physical shape from its stored coefficients
    mu = ds["mu"].values
    vh = ds["Vh"].values
    m_mean = ds["m_mean"].values
    b_mean = ds["b_mean"].values
    n_coord = sum(
        1 for name in ds.data_vars if str(name).startswith("pga_coef_")
    )
    coef = np.array([
        ds[f"pga_coef_{j}"].sel(foil_id=foil_id).item()
        for j in range(n_coord)
    ])
    ratio = ds["thickness_ratio"].sel(foil_id=foil_id).item()
    phys = reconstruct_phys_shape(mu, vh, m_mean, b_mean, coef, ratio)

    fig, ax = plt.subplots(figsize=(4, 1.6))
    ax.plot(phys[:, 0], phys[:, 1], color="black", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title(foil_id, fontsize=9)
    ax.axis("off")
    fig.tight_layout()

    out_path = f"{figures_dir}/{foil_id}/{foil_id}_shape.png"
    save_or_show(fig, out_path, dpi=150, bbox_inches="tight", quiet=True)


def plot_pga_pairs_worst(
    ds: xr.Dataset,
    fname: str = "output/figures/pga_pairs_worst.png",
    min_xfoil_conv: float = 0.75,
    min_foils: int = 3,
) -> None:
    """Corner plot of PGA coords + thickness ratio, foils below threshold."""
    from foilpolars.grassmann import _draw_pga_corner

    # Foils whose XFoil convergence fraction falls below the threshold,
    # same criterion as plot_convergence's threshold line
    xfoil_conv = ds["converged"].sel(fidelity="xfoil")
    foil_ids_all = ds["foil_id"].values
    xfoil_frac_all = xfoil_conv.mean(dim=("alpha", "Re", "n_crit")).values
    foil_ids = foil_ids_all[xfoil_frac_all < min_xfoil_conv]
    n = len(foil_ids)

    # Too few foils for a KDE corner plot (e.g. none when XFoil never
    # ran); skip instead of letting gaussian_kde fail on a tiny sample
    if n < min_foils:
        print(
            f"Skipping {fname}: only {n} foils below "
            f"{min_xfoil_conv:.2f} XFoil convergence (need "
            f">= {min_foils})"
        )
        return

    # Each selected foil's PGA coefficients + thickness ratio, no baseline
    # stars since only the perturbed shapes are of interest here
    n_coord = sum(
        1 for name in ds.data_vars if str(name).startswith("pga_coef_")
    )
    coefs = np.column_stack([
        ds[f"pga_coef_{j}"].sel(foil_id=foil_ids).values
        for j in range(n_coord)
    ])
    ratios = ds["thickness_ratio"].sel(foil_id=foil_ids).values
    sampled = np.column_stack([coefs, ratios])
    labels = [f"PGA {i + 1}" for i in range(n_coord)] + ["thickness ratio"]

    fig = _draw_pga_corner(
        sampled, labels,
        title=f"PGA pairs ({n} foils below {min_xfoil_conv:.2f} XFoil "
        "convergence)",
    )
    save_or_show(fig, fname, dpi=150, bbox_inches="tight")


def plot_foil_re_comparison(
    ds: xr.Dataset,
    foil_id: str,
    re: float,
    sigma: float,
    n_crit: float | None = None,
    figures_dir: str = "output/figures",
) -> None:
    """One (foil, Re, n_crit) figure: XFoil points vs NeuralFoil, 3x2."""
    if n_crit is None:
        n_crit = float(ds["n_crit"].values[0])
    sub = ds.sel(foil_id=foil_id, Re=re, n_crit=n_crit)
    alpha = sub["alpha"].values
    has_xfoil = "xfoil" in ds["fidelity"].values
    has_nf = "neuralfoil" in ds["fidelity"].values

    xf = sub.sel(fidelity="xfoil") if has_xfoil else None
    nf = sub.sel(fidelity="neuralfoil") if has_nf else None

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    ax_cl, ax_cd, ax_cm, ax_cpmin, ax_conf, ax_bkt = axes.flatten()

    # Quantities shared by both solvers: XFoil points, NeuralFoil line
    shared = [
        (ax_cl, "Cl", "$C_l$"), (ax_cd, "Cd", "$C_d$"),
        (ax_cm, "Cm", "$C_m$"),
    ]
    for ax, var, ylabel in shared:
        if has_xfoil:
            ax.plot(
                alpha, xf[var].values, "o", color="black", markersize=4,
                label="XFoil",
            )
        if has_nf:
            ax.plot(
                alpha, nf[var].values, color="tab:blue", lw=1.5,
                label="NeuralFoil",
            )
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.legend(fontsize=8)

    # Cp_min panel: plotted as -Cp_min (positive = stronger suction), same
    # sign convention as the cavitation bucket and the sigma threshold line
    if has_xfoil:
        ax_cpmin.plot(
            alpha, -xf["Cp_min"].values, "o", color="black", markersize=4,
            label="XFoil",
        )
    if has_nf:
        ax_cpmin.plot(
            alpha, -nf["Cp_min"].values, color="tab:blue", lw=1.5,
            label="NeuralFoil",
        )
    ax_cpmin.set_xlabel(r"$\alpha$ (deg)")
    ax_cpmin.set_ylabel("$-C_{p,\\min}$")
    ax_cpmin.grid(True, linewidth=0.4)
    ax_cpmin.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax_cpmin.legend(fontsize=8)

    # NeuralFoil analysis confidence, with XFoil convergence (1/0) overlaid
    if has_nf:
        ax_conf.plot(
            alpha, nf["analysis_confidence"].values, color="tab:blue",
            lw=1.5, label="NeuralFoil confidence",
        )
    if has_xfoil:
        ax_conf.plot(
            alpha, xf["converged"].values, "x", color="black",
            markersize=6, label="XFoil converged",
        )
    ax_conf.set_ylim(0, 1)
    ax_conf.set_xlabel(r"$\alpha$ (deg)")
    ax_conf.set_ylabel("Analysis confidence")
    ax_conf.grid(True, linewidth=0.4)
    ax_conf.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax_conf.legend(fontsize=8)

    # Cavitation bucket: Cl vs sigma_i = -Cp_min, both solvers as points/line
    if has_nf:
        nf_order = np.argsort(nf["Cl"].values)
        ax_bkt.plot(
            nf["Cl"].values[nf_order], -nf["Cp_min"].values[nf_order],
            color="tab:blue", lw=1.5, label="NeuralFoil",
        )
    if has_xfoil:
        ax_bkt.plot(
            xf["Cl"].values, -xf["Cp_min"].values, "o", color="black",
            markersize=4, label="XFoil",
        )
    ax_bkt.axhline(sigma, color="red", ls=":", lw=2.0)
    ax_bkt.set_xlabel("$C_l$")
    ax_bkt.set_ylabel("$\\sigma_i = -C_{p,\\min}$")
    ax_bkt.set_title("Cavitation bucket")
    ax_bkt.grid(True, linewidth=0.4)
    ax_bkt.legend(fontsize=8)

    fig.suptitle(
        f"{foil_id}  —  XFoil vs NeuralFoil  Re={re:.0e}  "
        f"N_crit={n_crit:g}",
        fontsize=13,
    )
    fig.tight_layout()

    re_tag = f"Re{re:.0e}".replace("+", "")
    fname_tag = f"Ncrit{n_crit:g}"
    out_path = (
        f"{figures_dir}/{foil_id}/{foil_id}_{re_tag}_{fname_tag}.png"
    )
    save_or_show(fig, out_path, dpi=150, quiet=True)
