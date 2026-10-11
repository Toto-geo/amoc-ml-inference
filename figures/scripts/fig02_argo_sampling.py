"""Figure 2: Argo sampling footprint and virtual-profile geometry at 34.5 S.

Two panels: (a) the real January-2013 Argo distribution inside the
reconstruction domain on an orthographic globe, and (b) the 3-D virtual
profile geometry with the Argo jitter footprint drawn at the real
sigma = 0.15 grid-cell scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_preparation/data"
ARGO = DATA / "input/virtual_argo/oras5_virtual_argo_density_input_34p5S_pm3deg_1958_2025.nc"
REAL = DATA / "argo/03_argo_real_coverage.csv"
TEMPERATURE = DATA / "dynamic/Temperature/oras5_votemper_34p5S_pm3deg_1958_2025.nc"
LON_MIN, LON_MAX = -51.5, 17.557666666666666    # reconstruction-domain longitudes
LAT_MIN, LAT_MAX = -37.5, -31.5                 # reconstruction-domain latitudes
JITTER_SIGMA_CELLS = 0.15                       # Argo jitter scale in grid-cell units


def data():
    """Load the January 2013 real-Argo and virtual-Argo coverage and the T-grid axes."""
    real = pd.read_csv(REAL, parse_dates=["time"])
    real = real[
        (real.time.dt.to_period("M") == pd.Period("2013-01"))
        & real.lon.between(LON_MIN, LON_MAX)
        & real.lat.between(LAT_MIN, LAT_MAX)
    ]
    with xr.open_dataset(ARGO) as ds:
        month = ds.isel(time=0).load()                 # single reference month (January 2013)
        z = ds.z.values.copy()
    with xr.open_dataset(TEMPERATURE) as ds:
        lon_grid = ds.nav_lon.values[0].copy()          # 1-D longitude axis of the T grid
        lat_grid = ds.nav_lat.values[:, 0].copy()       # 1-D latitude axis of the T grid
    valid = month.profile_mask.values.astype(bool)
    if len(real) != int(valid.sum()) or not len(real):
        raise ValueError("Real and virtual Argo coverage differ for January 2013")
    return real, month, z, lon_grid, lat_grid, valid


def draw(real, month, z, lon_grid, lat_grid, valid):
    """Compose the two-panel figure: Argo coverage globe and virtual-profile geometry."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(11.35, 5.55), dpi=150)
    globe = fig.add_axes([0.02, 0.14, 0.43, 0.79], projection=ccrs.Orthographic(-20, -20))
    section = fig.add_axes([0.44, 0.14, 0.54, 0.80], projection="3d")
    fig.text(0.20, 0.98, "(a) Argo distribution in reconstruction domain", ha="center", va="top", fontsize=15)
    fig.text(0.72, 0.98, "(b) Virtual profile geometry", ha="center", va="top", fontsize=15)

    # Panel (a): globe with land, ocean, the domain box, Argo dots and the 34.5 S line.
    globe.set_global()
    globe.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    globe.add_feature(cfeature.LAND, facecolor="#ece8d5", edgecolor="0.35", linewidth=0.5)
    globe.coastlines(color="0.35", linewidth=0.5)
    plate = ccrs.PlateCarree()
    # Four edges of the lon/lat domain rectangle, drawn with 200 segments each.
    edges = [
        (np.linspace(LON_MIN, LON_MAX, 200), np.full(200, LAT_MIN)),
        (np.full(200, LON_MAX), np.linspace(LAT_MIN, LAT_MAX, 200)),
        (np.linspace(LON_MAX, LON_MIN, 200), np.full(200, LAT_MAX)),
        (np.full(200, LON_MIN), np.linspace(LAT_MAX, LAT_MIN, 200)),
    ]
    for lon, lat in edges:
        globe.plot(lon, lat, color="#e5a12f", linewidth=1.0, transform=plate)
    globe.scatter(real.lon, real.lat, s=2, color="#2d9cdb", transform=plate, zorder=5)
    # Dashed 34.5 S reference line split around the domain. Densely sample
    # each segment so the projected parallel remains smooth on the globe.
    west_lon = np.linspace(-180.0, LON_MIN - 1.0, 720)
    east_lon = np.linspace(LON_MAX + 1.0, 180.0, 720)
    for lon in (west_lon, east_lon):
        globe.plot(
            lon,
            np.full_like(lon, -34.5),
            color="0.4",
            linewidth=1.0,
            linestyle=(0, (8.0, 2.2)),
            transform=plate,
            zorder=0.8,
        )
    globe.text(-72, -36.2, r"34.5$^\circ$S", color="0.4", fontsize=9,
               transform=plate, ha="center", va="center", zorder=8)

    # Panel (b): translucent domain floor at depth 0.
    domain = [[(LAT_MIN, LON_MIN, 0), (LAT_MAX, LON_MIN, 0),
               (LAT_MAX, LON_MAX, 0), (LAT_MIN, LON_MAX, 0)]]
    section.add_collection3d(Poly3DCollection(domain, facecolors=(0.96, 0.69, 0.25, 0.14),
                                              edgecolors="#e5a12f", linewidths=1.0))
    # Density range for the viridis_r colour map; profiles above 2000 m only.
    norm = Normalize(1025, 1037)
    cmap = cm.viridis_r
    lon = month.lon.values[valid]
    lat = month.lat.values[valid]
    rho = month.rho.values[valid]
    boxes = []
    for x, y, profile in zip(lat, lon, rho):
        good = np.isfinite(profile) & np.isfinite(z) & (z <= 2000)
        depth = z[good]
        if len(depth) < 2:
            continue
        # Each profile: a vertical polyline coloured by in-situ density on consecutive segments.
        coords = np.column_stack((np.full(len(depth), x), np.full(len(depth), y), depth))
        segments = np.stack((coords[:-1], coords[1:]), axis=1)
        lines = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=1.05, alpha=0.88)
        lines.set_array(0.5 * (profile[good][:-1] + profile[good][1:]))
        section.add_collection3d(lines)
        # Jitter footprint around each profile position: a square at depth 0, size +/- 1 sigma in grid cells.
        ix = np.clip(np.searchsorted(lon_grid, y), 1, len(lon_grid) - 1)
        iy = np.clip(np.searchsorted(lat_grid, x), 1, len(lat_grid) - 1)
        dx = JITTER_SIGMA_CELLS * (lon_grid[ix] - lon_grid[ix - 1])
        dy = JITTER_SIGMA_CELLS * (lat_grid[iy] - lat_grid[iy - 1])
        boxes.append([(x - dy, y - dx, 0), (x + dy, y - dx, 0),
                      (x + dy, y + dx, 0), (x - dy, y + dx, 0)])
    section.add_collection3d(Poly3DCollection(boxes, facecolors=(0.82, 0.12, 0.12, 0.58),
                                               edgecolors=(0.82, 0.12, 0.12, 0.98), linewidths=0.55))
    # Axis limits inverted so latitude decreases east-ward and depth increases downward.
    section.set(xlim=(LAT_MAX, LAT_MIN), ylim=(LON_MIN, LON_MAX), zlim=(2000, 0),
                xlabel="Latitude", ylabel="Longitude", zlabel="Depth (m)")
    section.set_xticks([-37.5, -34.5, -31.5])
    section.set_yticks([-50, -30, -10, 10])
    section.set_zticks([0, 1000, 2000])
    section.tick_params(labelsize=9, pad=0)
    section.view_init(elev=24, azim=-58)
    section.set_box_aspect((1.25, 2.75, 0.82), zoom=1.04)
    # Shared colour bar for density.
    scale = cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(scale, cax=fig.add_axes([0.64, 0.075, 0.34, 0.04]), orientation="horizontal")
    bar.set_label(r"In situ density $\rho$ (kg m$^{-3}$)", labelpad=4)
    bar.ax.xaxis.set_label_position("top")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color="#2d9cdb", label="Argo profile"),
                        Patch(facecolor=(0.82, 0.12, 0.12, 0.58), label=r"Jitter scale ($\pm 1\sigma$)")],
               loc="lower center", bbox_to_anchor=(0.43, 0.005), ncol=2, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw(*data())
    output = args.output_dir / "fig02_argo_sampling.pdf"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
