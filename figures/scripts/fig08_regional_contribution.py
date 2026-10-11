"""Figure 8: two-globe longitude-bin permutation sensitivity.

Two orthographic globes (one per regime) show the mean MAE increase caused
by permuting a single longitude bin across all four surface fields. Each bin
is drawn as a curved rectangle along 34.5 S, coloured by the sensitivity on
a square-root power scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY = (ROOT / "data_preparation/data/input/SSH"
                    / "oras5_ssh_input_AZ_20bin_means_1958_2025.nc")
LAT_TARGET = -34.5
LAT_MIN, LAT_MAX = -36.5, -32.5                # band width of the drawn rectangles
EXPERIMENTS = (("M", "a) Monthly reconstruction"),
               ("Q", "b) Quarterly reconstruction"))


def plot_data(summary: pd.DataFrame, geometry: Path) -> pd.DataFrame:
    """Attach bin-geometry columns to the 20-bin bin-permutation summaries."""
    required = {"regime", "bins", "kind", "bin_index", "mean_pct"}
    if required - set(summary):
        raise ValueError(f"Missing permutation-summary columns: {sorted(required - set(summary))}")
    selected = summary.loc[summary.kind.eq("bin") & summary.bins.eq(20)].copy()
    if (len(selected) != 40 or selected.duplicated(["regime", "bin_index"]).any()
            or any(set(group.bin_index) != set(range(1, 21))
                   for _, group in selected.groupby("regime"))
            or set(selected.regime) != {"M", "Q"}
            or not np.isfinite(selected.mean_pct).all()):
        raise ValueError("Expected 20 finite longitude-bin summaries in each final regime")
    # Read the bin edges and centres from the SSH 20-bin product.
    with xr.open_dataset(geometry) as dataset:
        if int(dataset.attrs.get("n_bins", -1)) != 20:
            raise ValueError("Expected 20-bin SSH geometry")
        left = dataset.bin_edge_left.values.astype(float)
        right = dataset.bin_edge_right.values.astype(float)
        center = dataset.bin_lon_center.values.astype(float)
    if (any(len(edges) != 20 for edges in (left, right, center))
            or not np.all(left < right)
            or not np.allclose(right[:-1], left[1:], rtol=0, atol=1e-4)
            or not np.allclose((left + right) / 2, center, rtol=0, atol=1e-4)):
        raise ValueError("Invalid surface-bin geometry")
    geometry_frame = pd.DataFrame({"bin_index": np.arange(1, 21), "lon_left": left,
                                   "lon_right": right, "lon_center": center})
    selected = selected.merge(geometry_frame, on="bin_index", validate="many_to_one")
    selected["mae_increase_pct_clipped"] = selected.mean_pct.clip(lower=0.0)       # negatives shown as 0
    return selected.sort_values(["regime", "bin_index"]).reset_index(drop=True)


def build_figure(data: pd.DataFrame):
    """Compose the two-globe figure with a shared square-root power-scale colour bar."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 600,
                         "font.size": 9.5, "axes.titlesize": 11.0,
                         "axes.labelsize": 10.0, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5})
    projection = ccrs.Orthographic(-20, -20)
    data_crs = ccrs.PlateCarree()
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "light_orange_to_red",
        ["#fff7ec", "#fee8c8", "#fdbb84", "#fc8d59", "#d7301f", "#7f0000"])
    vmax = np.ceil(float(data.mae_increase_pct_clipped.max()) * 10) / 10
    if vmax <= 0:
        vmax = 1.0
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)   # sqrt scale emphasises small values
    fig = plt.figure(figsize=(10.0, 6.05))
    axes = [fig.add_axes([-0.040, 0.170, 0.44, 0.760], projection=projection),
            fig.add_axes([0.600, 0.170, 0.44, 0.760], projection=projection)]
    for ax, (regime, _) in zip(axes, EXPERIMENTS):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#d9d4c0", edgecolor="gray",
                       linewidth=0.7, zorder=1)
        ax.coastlines(color="gray", linewidth=0.7, zorder=2)
        ax.spines["geo"].set_edgecolor("0.25")
        ax.spines["geo"].set_linewidth(0.8)
        # Dashed 34.5 S latitude line and label.
        lon_line = np.linspace(-180.0, 180.0, 1440)
        ax.plot(lon_line, np.full_like(lon_line, LAT_TARGET), color="0.62",
                linewidth=1.0, linestyle=(0, (8.0, 2.2)),
                transform=data_crs, zorder=0.8)
        ax.text(-88.0, LAT_TARGET + 4.0, "34.5°S", transform=data_crs,
                ha="center", va="center", fontsize=8.2, color="#666666", zorder=8)
        # One curved rectangle per bin, coloured by its clipped MAE increase.
        for _, row in data.loc[data.regime.eq(regime)].iterrows():
            lon_top = np.linspace(row.lon_left, row.lon_right, 220)
            lon_bottom = np.linspace(row.lon_right, row.lon_left, 220)
            ax.fill(np.concatenate([lon_top, lon_bottom]),
                    np.concatenate([np.full_like(lon_top, LAT_MAX),
                                    np.full_like(lon_bottom, LAT_MIN)]),
                    facecolor=cmap(norm(row.mae_increase_pct_clipped)),
                    edgecolor="black", linewidth=1.15, transform=data_crs,
                    zorder=7, antialiased=True)
    fig.text(0.180, 0.985, EXPERIMENTS[0][1], ha="center", va="top",
             fontsize=19.9, color="0.05")
    fig.text(0.820, 0.985, EXPERIMENTS[1][1], ha="center", va="top",
             fontsize=19.9, color="0.05")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes([0.18, 0.065, 0.64, 0.035])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Mean MAE increase after bin permutation [%]",
                       fontsize=14.0, labelpad=5)
    colorbar.ax.tick_params(labelsize=11.2, length=3.0, width=0.8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-summary", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = plot_data(pd.read_csv(args.permutation_summary), args.geometry)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    fig = build_figure(data)
    fig.savefig(args.output_dir / "fig08_regional_contribution.pdf",
                bbox_inches="tight")
    data.to_csv(args.output_dir / "regional_plot_data.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
