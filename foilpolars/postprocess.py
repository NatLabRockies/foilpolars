"""Post-processing utilities: cavitation proxy, convergence summary, plots."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foilpolars.utils import save_or_show


def compute_cavitation_sigma(
    re_values: list[float], chord: float, depth: float, temperature: float,
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


def save_full_results(ds: xr.Dataset, out_path: str) -> None:
    """Write every (foil_id, alpha, Re, fidelity) row, no summarising."""
    # Flatten to one row per (foil_id, alpha, Re, fidelity) and write csv
    df = ds.to_dataframe().reset_index()
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_confidence_map(ds: xr.Dataset) -> None:
    """Heatmap of NeuralFoil analysis_confidence over (alpha, foil_id)."""
    if "analysis_confidence" not in ds.data_vars:
        print("analysis_confidence not in dataset; skipping.")
        return
    if "neuralfoil" not in ds["fidelity"].values:
        print("neuralfoil fidelity not in dataset; skipping.")
        return

    # Use middle Re and n_crit values
    re_values = ds["Re"].values
    re_mid = re_values[len(re_values) // 2]
    n_crit_mid = ds["n_crit"].values[len(ds["n_crit"]) // 2]

    data = (
        ds["analysis_confidence"]
        .sel(Re=re_mid, n_crit=n_crit_mid, fidelity="neuralfoil")
        .values
    )  # shape: (n_foils, n_alpha)

    foil_ids = ds["foil_id"].values
    alphas = ds["alpha"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        alphas,
        np.arange(len(foil_ids)),
        data,
        cmap="RdYlGn",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Analysis confidence")
    ax.set_yticks(np.arange(len(foil_ids)))
    ax.set_yticklabels(foil_ids, fontsize=8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_title(
        f"NeuralFoil analysis confidence map  Re={re_mid:.0e}"
    )
    fig.tight_layout()

    fname = "output/figures/confidence_map_neuralfoil.png"
    save_or_show(fig, fname, dpi=150)


def plot_foil_re_comparison(
    ds: xr.Dataset, foil_id: str, re: float, sigma: float,
    n_crit: float | None = None, figures_dir: str = "output/figures",
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
    save_or_show(fig, out_path, dpi=150)
