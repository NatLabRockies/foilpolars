"""Grassmannian shape parameterization of raw airfoil coordinates (G2Aero)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from g2aero.Grassmann import (
    PGA,
    Karcher,
    landmark_affine_transform,
    perturb_gr_shape,
)
from g2aero.utils import check_selfintersect
from scipy.stats import gaussian_kde
from tqdm import tqdm

from foilpolars.utils import save_or_show

# Shared palette so baseline/perturbed/mean read the same way across
# every plotting function in this module
BASELINE_COLOR = "k"
PERTURBED_COLOR = "grey"
MEAN_COLOR = "tab:orange"


def compute_grassmann(
    shapes: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Map each foil's raw (x/c, y/c) coordinates onto the Grassmannian."""
    print("[compute_grassmann] starting")
    results: dict[str, dict[str, np.ndarray]] = {}

    # Landmark-affine standardize each foil independently, since raw
    # foils may have different point counts
    for desig, coords in shapes.items():
        x_gr, m, b = landmark_affine_transform(coords)
        results[desig] = {"X_gr": x_gr, "M": m, "b": b}

    return results


def save_grassmann_baseline(
    results: dict[str, dict[str, np.ndarray]],
    out_path: str = "output/data/foil_baseline_grass.nc",
) -> None:
    """Write each foil's Grassmann coords + affine transform to one netcdf."""
    from foilpolars.shapes import save_shapes

    # Grassmann coords reuse the physical-shape netcdf writer; M/b (the
    # affine transform back to physical coords) ride along as extras
    x_gr = {desig: r["X_gr"] for desig, r in results.items()}
    extra_columns = {
        desig: dict(zip(
            ["M_00", "M_01", "M_10", "M_11", "b_0", "b_1"],
            [*r["M"].ravel(), *r["b"]],
        ))
        for desig, r in results.items()
    }
    save_shapes(
        x_gr, out_path=out_path, columns=("Xgr_0", "Xgr_1"),
        extra_columns=extra_columns,
    )


def save_grassmann_cache(
    path: str,
    repaneled_shapes: dict[str, np.ndarray],
    basis: dict[str, object],
    perturbed: dict[str, np.ndarray],
) -> None:
    """Cache repaneled shapes, PGA basis and perturbed samples for replot."""
    # A single npz beside the sweep output lets `plot` redraw every
    # Grassmann figure without repeating the PGA/Karcher computation
    foil_ids = list(repaneled_shapes.keys())
    np.savez(
        path,
        shape_foil_ids=np.array(foil_ids),
        shapes=np.stack([repaneled_shapes[i] for i in foil_ids]),
        basis_foil_ids=np.array(basis["foil_ids"]),
        basis_X_gr=basis["X_gr"], basis_M=basis["M"], basis_b=basis["b"],
        basis_mu=basis["mu"], basis_Vh=basis["Vh"], basis_t=basis["t"],
        perturbed_X_gr=perturbed["X_gr"], perturbed_phys=perturbed["phys"],
        perturbed_coef=perturbed["coef"],
        perturbed_thickness_ratio=perturbed["thickness_ratio"],
    )


def load_grassmann_cache(
    path: str,
) -> tuple[
    dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray],
]:
    """Reload the cached shapes, PGA basis and perturbed samples."""
    data = np.load(path)
    repaneled_shapes = dict(zip(data["shape_foil_ids"], data["shapes"]))
    basis = {
        "foil_ids": list(data["basis_foil_ids"]), "X_gr": data["basis_X_gr"],
        "M": data["basis_M"], "b": data["basis_b"], "mu": data["basis_mu"],
        "Vh": data["basis_Vh"], "t": data["basis_t"],
    }
    perturbed = {
        "X_gr": data["perturbed_X_gr"], "phys": data["perturbed_phys"],
        "coef": data["perturbed_coef"],
        "thickness_ratio": data["perturbed_thickness_ratio"],
    }
    return repaneled_shapes, basis, perturbed


def shapes_dict(
    coords: np.ndarray,
    prefix: str = "p",
    width: int = 4,
) -> dict[str, np.ndarray]:
    """Assign each sample in a stacked coordinate array a sequential id."""
    return {
        f"{prefix}{j:0{width}d}": sample for j, sample in enumerate(coords)
    }


def check_reconstruction(
    shapes: dict[str, np.ndarray],
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Print the max reconstruction error per foil as a sanity check."""
    # Reproject each foil's Grassmann coords through its affine transform
    # and compare against the original physical coordinates
    for desig, coords in shapes.items():
        r = results[desig]
        recon = r["X_gr"] @ r["M"] + r["b"]
        err = np.max(np.abs(recon - coords))
        print(f"{desig}: max reconstruction error = {err:.2e}")


def compute_pga_basis(
    shapes: dict[str, np.ndarray],
    n_coord: int = 4,
) -> dict[str, object]:
    """Batch-align all foils, then compute their Karcher mean + PGA basis."""
    print("[compute_pga_basis] starting")
    # Stack every foil's coordinates so they can be batch-aligned together
    foil_ids = list(shapes.keys())
    stacked = np.stack([shapes[desig] for desig in foil_ids])

    # Batched mapping Procrustes-aligns all shapes before the Karcher
    # mean and PGA are computed, so geodesic distances stay meaningful
    x_gr, m, b = landmark_affine_transform(stacked)
    mu = Karcher(x_gr)
    vh, _, t = PGA(mu, x_gr, n_coord=n_coord)

    return {
        "foil_ids": foil_ids, "X_gr": x_gr, "M": m, "b": b,
        "mu": mu, "Vh": vh, "t": t,
    }


def _draw_pga_corner(
    sampled: np.ndarray,
    labels: list[str],
    baseline: np.ndarray | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Corner (staircase) plot of sample density, baseline stars optional."""
    n_param = len(labels)
    grid_size = 60

    # Staircase (corner) layout: diagonal holds marginal histograms,
    # lower triangle holds pairwise KDE contours, upper triangle blank
    fig, axes = plt.subplots(
        n_param, n_param, figsize=(1.8 * n_param, 1.8 * n_param),
    )
    contour = None
    for i in range(n_param):
        for j in range(n_param):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(
                    sampled[:, i], bins=15, color=PERTURBED_COLOR, alpha=0.6,
                    density=True,
                )
                ax.set_yticks([])
            else:
                # Gaussian KDE on a grid, drawn as filled contours so
                # the sample cloud reads as a smooth density surface
                x, y = sampled[:, j], sampled[:, i]
                kde = gaussian_kde(np.vstack([x, y]))
                xi = np.linspace(x.min(), x.max(), grid_size)
                yi = np.linspace(y.min(), y.max(), grid_size)
                xx, yy = np.meshgrid(xi, yi)
                zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
                contour = ax.contourf(
                    xx, yy, zz.reshape(xx.shape), levels=12, cmap="viridis",
                )
                if baseline is not None:
                    ax.scatter(
                        baseline[:, j], baseline[:, i], marker="*", s=60,
                        color=BASELINE_COLOR, edgecolor="k", linewidth=0.5,
                        zorder=2, label="baseline foils",
                    )
                ax.grid(True, linewidth=0.3)
            if i == n_param - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

    # Baseline legend and density colorbar go inside the blank
    # upper-right triangle instead of outside the axes grid
    if baseline is not None:
        handles, legend_labels = axes[1, 0].get_legend_handles_labels()
        axes[0, n_param - 1].legend(
            handles, legend_labels, loc="center", fontsize=10,
        )
    if contour is not None and n_param > 2:
        # A fresh, narrower axes instead of the blank grid cell itself,
        # since that cell's ticks/labels were already switched off above
        pos = axes[1, n_param - 1].get_position()
        cax = fig.add_axes((
            pos.x0 + pos.width * 0.35, pos.y0, pos.width * 0.3, pos.height,
        ))
        fig.colorbar(contour, cax=cax, label="density")
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_pga_pairs(
    basis: dict[str, object],
    perturbed: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Corner plot of the 4 PGA coordinates plus the thickness ratio."""
    t = basis["t"]
    n_coord = t.shape[1]

    # Thickness ratio (5th param) is the baseline's counterpart to
    # thickness_ratio: ratio of M's two singular values (chord, thickness)
    singular_values = np.linalg.svd(basis["M"], compute_uv=False)
    baseline_ratio = singular_values[:, 1] / singular_values[:, 0]
    baseline = np.column_stack([t, baseline_ratio])
    sampled = np.column_stack(
        [perturbed["coef"], perturbed["thickness_ratio"]],
    )
    labels = [f"PGA {i + 1}" for i in range(n_coord)] + ["thickness ratio"]

    fig = _draw_pga_corner(sampled, labels, baseline=baseline)
    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")


def _normalize_le_te(phys: np.ndarray) -> np.ndarray:
    """Translate/rotate/scale so the LE is at (0, 0) and TE at (1, 0)."""
    # TE is the midpoint of the first/last points; LE is the point
    # farthest from it, matching aerosandbox's Airfoil.normalize()
    x_te, y_te = np.mean(phys[[0, -1]], axis=0)
    dist_to_te = np.hypot(phys[:, 0] - x_te, phys[:, 1] - y_te)
    le_index = np.argmax(dist_to_te)

    # Translate LE to the origin, scale so chord length is 1
    shifted = phys - phys[le_index]
    scale = 1.0 / dist_to_te[le_index]
    scaled = shifted * scale

    # Rotate about the origin so the TE lands on the x-axis at (1, 0)
    x_te, y_te = np.mean(scaled[[0, -1]], axis=0)
    angle = -np.arctan2(y_te, x_te)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = scaled @ rotation.T
    return _snap_te_to_chord(rotated)


def _snap_te_to_chord(phys: np.ndarray) -> np.ndarray:
    """Slide each TE surface point to x/c = 1 along its own local slope."""
    # Rotation only pins the TE midpoint, leaving upper/lower TE points
    # straddling x/c=1; extend/trim each along its last two TE points
    out = phys.copy()
    for te_idx, next_idx in ((0, 1), (-1, -2)):
        x0, y0 = phys[te_idx]
        x1, y1 = phys[next_idx]
        if x1 != x0:
            t = (1.0 - x0) / (x1 - x0)
            out[te_idx] = [1.0, y0 + t * (y1 - y0)]
    return out


def perturb_grassmann(
    basis: dict[str, object],
    n_perturb: int = 20,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample new shapes around the Karcher mean, as in g2aero's PGA space."""
    rng = np.random.default_rng(seed)
    mu, vh, t = basis["mu"], basis["Vh"], basis["t"]
    m_mean, b_mean = np.mean(basis["M"], axis=0), np.mean(basis["b"], axis=0)

    # M's chord/thickness singular-value ratio (dropped by compute_pga_basis's
    # mean-M) is sampled alongside the PGA coordinates, not a separate stage
    singular_values = np.linalg.svd(basis["M"], compute_uv=False)
    ratio_min = np.min(singular_values[:, 1] / singular_values[:, 0])
    ratio_max = np.max(singular_values[:, 1] / singular_values[:, 0])
    u_mean, d_mean, vh_mean = np.linalg.svd(m_mean)

    # One joint 5-D box: 4 PGA coords (per-mode range, matching
    # g2aero's sample_coef) plus thickness ratio, drawn together
    axis_min = np.append(np.min(t, axis=0), ratio_min)
    axis_max = np.append(np.max(t, axis=0), ratio_max)

    # Exp-map to the manifold, then to x/y via the mean affine transform;
    # renormalize LE/TE and resample on self-intersection, as g2aero does
    gr_shapes = np.empty((n_perturb, mu.shape[0], 2))
    coords = np.empty((n_perturb, mu.shape[0], 2))
    coefs = np.empty((n_perturb, t.shape[1]))
    ratios = np.empty(n_perturb)
    for j in tqdm(range(n_perturb), desc="[perturb_grassmann] sampling"):
        while True:
            sample = rng.uniform(axis_min, axis_max)
            coef, ratio = sample[:-1], sample[-1]
            m_sample = (
                u_mean @ np.diag([d_mean[0], d_mean[0] * ratio]) @ vh_mean
            )
            gr_shape = perturb_gr_shape(vh, mu, coef)
            phys = gr_shape @ m_sample + b_mean
            phys = _normalize_le_te(phys)
            if not check_selfintersect(phys):
                break
        gr_shapes[j] = gr_shape
        coords[j], coefs[j], ratios[j] = phys, coef, ratio

    return {
        "X_gr": gr_shapes, "phys": coords, "coef": coefs,
        "thickness_ratio": ratios,
    }


def reconstruct_phys_shape(
    mu: np.ndarray,
    vh: np.ndarray,
    m_mean: np.ndarray,
    b_mean: np.ndarray,
    coef: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Rebuild one perturbed shape's (x/c, y/c) from its saved PGA params."""
    # Same exp-map + affine reconstruction as perturb_grassmann's
    # sampling step, replayed for one saved (coef, ratio) pair
    u_mean, d_mean, vh_mean = np.linalg.svd(m_mean)
    m_sample = u_mean @ np.diag([d_mean[0], d_mean[0] * ratio]) @ vh_mean
    gr_shape = perturb_gr_shape(vh, mu, coef)
    phys = gr_shape @ m_sample + b_mean
    return _normalize_le_te(phys)


def add_pga_columns(
    ds: xr.Dataset,
    perturbed: dict[str, np.ndarray],
) -> xr.Dataset:
    """Attach each shape's PGA coefs, SVD thickness ratio and t/c_max."""
    coef = perturbed["coef"]
    coef_vars = {
        f"pga_coef_{i}": (("foil_id",), coef[:, i])
        for i in range(coef.shape[1])
    }
    return ds.assign(
        thickness_ratio=(("foil_id",), perturbed["thickness_ratio"]),
        thickness_max=(("foil_id",), _max_thickness(perturbed["phys"])),
        **coef_vars,
    )


def add_shared_basis_params(
    ds: xr.Dataset,
    basis: dict[str, object],
) -> xr.Dataset:
    """Attach the Karcher mean, PGA basis and mean affine transform."""
    m_mean = np.mean(basis["M"], axis=0)
    b_mean = np.mean(basis["b"], axis=0)

    return ds.assign(
        mu=(("point", "xy"), basis["mu"]),
        Vh=(("mode", "point_xy_flat"), basis["Vh"]),
        m_mean=(("xy", "xy2"), m_mean),
        b_mean=(("xy",), b_mean),
    )


def _draw_physical_baseline(
    ax: plt.Axes,
    shapes: dict[str, np.ndarray],
    karcher_phys: np.ndarray,
) -> None:
    """Plot dataset foils black, with the Karcher mean bold orange on top."""
    for coords in shapes.values():
        ax.plot(
            coords[:, 0], coords[:, 1], color=BASELINE_COLOR, linewidth=0.5,
        )
    ax.plot(
        karcher_phys[:, 0], karcher_phys[:, 1], color=MEAN_COLOR,
        linewidth=2, label="Karcher mean",
    )
    ax.legend(fontsize=8)


def _draw_physical_samples(
    ax: plt.Axes,
    perturbed: dict[str, np.ndarray],
    karcher_phys: np.ndarray,
) -> None:
    """Plot perturbed samples as thin grey lines, with the Karcher mean."""
    samples = perturbed["phys"]
    for sample in samples:
        ax.plot(
            sample[:, 0], sample[:, 1], color=PERTURBED_COLOR, linewidth=0.5,
            alpha=0.5,
        )
    ax.plot(
        karcher_phys[:, 0], karcher_phys[:, 1], color=MEAN_COLOR,
        linewidth=2, label="Karcher mean",
    )
    ax.legend(fontsize=8)


def plot_perturbed_shapes(
    shapes: dict[str, np.ndarray],
    basis: dict[str, object],
    perturbed: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Plot physical baseline/perturbed foils, one full-width row each."""
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 3.5), constrained_layout=True,
    )

    m_mean = np.mean(basis["M"], axis=0)
    b_mean = np.mean(basis["b"], axis=0)
    karcher_phys = basis["mu"] @ m_mean + b_mean

    # Baseline row, then perturbed row, each full-width with a true
    # (equal) aspect ratio so the thin foil profiles read correctly
    _draw_physical_baseline(axes[0], shapes, karcher_phys)
    _draw_physical_samples(axes[1], perturbed, karcher_phys)
    for ax in axes:
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(True, linewidth=0.3)

    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")


def _te_thickness(coords: np.ndarray) -> np.ndarray:
    """Gap between each foil's upper/lower TE points (index 0 and -1)."""
    return np.abs(coords[..., 0, 1] - coords[..., -1, 1])


def _max_thickness(coords: np.ndarray) -> np.ndarray:
    """Per-shape max upper/lower surface separation (t/c), via aerosandbox."""
    import aerosandbox as asb

    return np.array([
        asb.Airfoil(coordinates=c).max_thickness() for c in coords
    ])


def plot_te_thickness_histogram(
    perturbed: dict[str, np.ndarray],
    save_path: str | None = None,
    min_te_thickness: float = 1e-4,
) -> None:
    """Histogram of perturbed TE thickness (y/c), flags shapes below min."""
    # Trailing edge thickness per perturbed sample, in physical
    # (x/c, y/c) coordinates
    perturbed_te = _te_thickness(perturbed["phys"])

    # Log-spaced bins so equal-width bars on the log x-axis still
    # resolve the low-thickness end of the distribution
    log_bins = np.logspace(
        np.log10(perturbed_te.min()), np.log10(perturbed_te.max()), 100,
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(perturbed_te, bins=log_bins, color="tab:blue", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("trailing-edge thickness (y/c)")
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.3)

    # Min-thickness line, plus a count/id readout of foils under it
    # (same p#### ids `shapes_dict` assigns this array in the sweep)
    below = np.nonzero(perturbed_te < min_te_thickness)[0]
    foil_ids = list(shapes_dict(perturbed["phys"]).keys())
    below_ids = [foil_ids[i] for i in below]
    ax.axvline(
        min_te_thickness, color="tab:red", linestyle="--", linewidth=1,
        label=f"min allowed = {min_te_thickness:.0e}",
    )
    ax.text(
        0.01, 0.92,
        f"{len(below)}/{len(perturbed_te)} foils below "
        f"{min_te_thickness:.0e}",
        transform=ax.transAxes, fontsize=8, color="tab:red",
    )
    ax.legend(fontsize=8)
    if below_ids:
        print(
            f"{len(below_ids)} foils below {min_te_thickness:.0e} TE "
            f"thickness: {', '.join(below_ids)}"
        )
    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")


def plot_max_thickness_histogram(
    perturbed: dict[str, np.ndarray],
    save_path: str | None = None,
    min_thickness: float = 0.05,
) -> None:
    """Histogram of perturbed max thickness (t/c), flags shapes below min."""
    # Max upper/lower surface separation per perturbed sample, in
    # physical (x/c, y/c) coordinates
    perturbed_max_t = _max_thickness(perturbed["phys"])

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(perturbed_max_t, bins=30, color="tab:blue", alpha=0.7)
    ax.set_xlabel("max thickness (t/c)")
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.3)

    # Min-thickness line, plus a count/id readout of foils under it
    # (same p#### ids `shapes_dict` assigns this array in the sweep)
    below = np.nonzero(perturbed_max_t < min_thickness)[0]
    foil_ids = list(shapes_dict(perturbed["phys"]).keys())
    below_ids = [foil_ids[i] for i in below]
    ax.axvline(
        min_thickness, color="tab:red", linestyle="--", linewidth=1,
        label=f"min allowed = {min_thickness:.2f}",
    )
    ax.text(
        0.01, 0.92,
        f"{len(below)}/{len(perturbed_max_t)} foils below "
        f"{min_thickness:.2f}",
        transform=ax.transAxes, fontsize=8, color="tab:red",
    )
    ax.legend(fontsize=8)
    if below_ids:
        print(
            f"{len(below_ids)} foils below {min_thickness:.2f} max "
            f"thickness: {', '.join(below_ids)}"
        )
    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")


def plot_grassmann_baseline_samples(
    basis: dict[str, object],
    perturbed: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Plot Grassmann baseline/perturbed shapes, one square row each."""
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 3.5), constrained_layout=True,
    )

    _draw_grassmann_baseline(axes[0], basis)
    _draw_grassmann_samples(axes[1], perturbed, mu=basis["mu"])

    save_or_show(fig, save_path, dpi=150, bbox_inches="tight")


def _draw_grassmann_baseline(
    ax: plt.Axes,
    basis: dict[str, object],
) -> None:
    """Plot every foil's baseline Grassmann representation on one axes."""
    # All baseline foils share one color/symbol and one legend entry,
    # since individual foil identity isn't of interest here
    for i, x_gr in enumerate(basis["X_gr"]):
        ax.plot(
            x_gr[:, 0], x_gr[:, 1], color=BASELINE_COLOR, linewidth=0.5,
            label="baseline" if i == 0 else None,
        )
    mu = basis["mu"]
    ax.plot(
        mu[:, 0], mu[:, 1], color=MEAN_COLOR, linewidth=2.5, zorder=3,
        label="Karcher mean",
    )

    ax.set_xlabel("X_gr[:, 0]")
    ax.set_ylabel("X_gr[:, 1]")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)


def _draw_grassmann_samples(
    ax: plt.Axes,
    perturbed: dict[str, np.ndarray],
    mu: np.ndarray,
) -> None:
    """Plot every perturbed sample's Grassmann representation on one axes."""
    samples = perturbed["X_gr"]
    for i, sample in enumerate(samples):
        ax.plot(
            sample[:, 0], sample[:, 1], color=PERTURBED_COLOR, linewidth=0.5,
            alpha=0.5, label="perturbed" if i == 0 else None,
        )
    ax.plot(
        mu[:, 0], mu[:, 1], color=MEAN_COLOR, linewidth=2.5, zorder=3,
        label="Karcher mean",
    )

    ax.set_xlabel("X_gr[:, 0]")
    ax.set_ylabel("X_gr[:, 1]")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)
