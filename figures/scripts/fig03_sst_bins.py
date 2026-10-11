#!/usr/bin/env python
# coding: utf-8
"""Figure 3: ORAS5 native SST and its 20-bin model-input representation.

Two orthographic globe panels show the January 2024 SST field inside the
reconstruction band and the matching 20-bin zonal means that enter the model.
The vector of bin means is drawn as curved coloured boxes along 34.5 S.
"""

# --- Repo bootstrap and runtime cache ---

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data_preparation/data"
ORAS5_DYNAMIC_DIR = DATA / "dynamic"
ORAS5_INPUT_DIR = DATA / "input"
parser = argparse.ArgumentParser(description="Recreate paper Figure 3: SST bin construction")
parser.add_argument("--output-dir", type=Path, required=True)
OUTPUT_DIR = parser.parse_args().output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Load ORAS5 native SST and 20-bin ML input ---

import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message="Engine 'argo' loading failed",
    category=RuntimeWarning,
)

TARGET_MONTH = "2024-01"      # month shown in both panels
N_BINS = 20                   # bin count for the 20-bin ML input

# Geographic box used by the SST input construction.
LON_MIN, LON_MAX = -51.5, 17.6
LAT_MIN, LAT_MAX = -36.5, -32.5
LAT_TARGET = 0.5 * (LAT_MIN + LAT_MAX)

FILE_NATIVE = ORAS5_DYNAMIC_DIR / "SST" / "oras5_sosstsst_34p5S_pm2deg_1958_2025.nc"
FILE_INPUT_20 = ORAS5_INPUT_DIR / "SST" / "oras5_sst_input_AZ_20bin_means_1958_2025.nc"


def select_month(da: xr.DataArray, time_name: str, target_month: str) -> xr.DataArray:
    """Return the single month of da, or its mean if several values share the target month."""
    time_vals = pd.to_datetime(da[time_name].values)
    target = pd.Period(target_month, freq="M")
    mask = (time_vals.year == target.year) & (time_vals.month == target.month)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No data found for {target_month}.")
    if len(indices) == 1:
        return da.isel({time_name: int(indices[0])})
    return da.isel({time_name: indices}).mean(time_name, skipna=True)


ds_native = xr.open_dataset(FILE_NATIVE)
ds_input = xr.open_dataset(FILE_INPUT_20)

sst_oras = select_month(ds_native["sosstsst"], "time_counter", TARGET_MONTH)
lon2d = ds_native["nav_lon"]
lat2d = ds_native["nav_lat"]

bin_means_da = select_month(ds_input["sst_bin_means"], "time_counter", TARGET_MONTH)
bin_means_sst = np.asarray(bin_means_da.values, dtype=float).reshape(-1)

bin_left = np.asarray(ds_input["bin_edge_left"].values, dtype=float)
bin_right = np.asarray(ds_input["bin_edge_right"].values, dtype=float)
bin_edges = np.concatenate([bin_left, [bin_right[-1]]])
bin_centers = 0.5 * (bin_left + bin_right)

if bin_means_sst.size != N_BINS:
    raise ValueError(f"Expected {N_BINS} SST bins, got {bin_means_sst.size}.")
if bin_edges.size != N_BINS + 1:
    raise ValueError(f"Expected {N_BINS + 1} bin edges, got {bin_edges.size}.")

target_label = pd.Period(TARGET_MONTH, freq="M").strftime("%b %Y")
sst_units = ds_input["sst_bin_means"].attrs.get("units", "degC")

print("=" * 100)
print("ORAS5 SST 20-BIN INPUT")
print("=" * 100)
print(f"Target month      : {target_label}")
print(f"Native file       : {FILE_NATIVE}")
print(f"20-bin input file : {FILE_INPUT_20}")
print(f"Native shape      : {sst_oras.shape}")
print(f"n bins            : {bin_means_sst.size}")
print(f"Bin longitude span: {bin_edges[0]:.3f} to {bin_edges[-1]:.3f}")
print(f"Units             : {sst_units}")




# --- Plot helpers and style ---

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path
from matplotlib.patches import PathPatch

GLOBE_PROJECTION = ccrs.Orthographic(-20, -20)
DATA_PROJECTION = ccrs.PlateCarree()
LAND_FACE = "#d9d4c0"
COAST_COLOR = "gray"
BOX_COLOR = "#666666"
BIN_EDGE_COLOR = "#666666"
LATITUDE_LINE_COLOR = BIN_EDGE_COLOR
SST_CMAP = plt.get_cmap("viridis")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 14.2,
    "xtick.labelsize": 11.8,
    "ytick.labelsize": 11.8,
    "legend.fontsize": 13.4,
})


def robust_norm(values, lower=2, upper=98):
    """Build a Normalize using the 2nd and 98th percentiles so outliers do not stretch the scale."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot build color normalization from all-NaN values.")

    vmin, vmax = np.percentile(finite, [lower, upper])
    if np.isclose(vmin, vmax):
        pad = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        vmin -= pad
        vmax += pad
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def make_box_path(lon_min, lon_max, lat_min, lat_max, n_lon=300, n_lat=160):
    """Return a Matplotlib Path tracing the four edges of a lon/lat rectangle."""
    lon_top = np.linspace(lon_min, lon_max, n_lon)
    lat_top = np.full_like(lon_top, lat_max)

    lat_right = np.linspace(lat_max, lat_min, n_lat)
    lon_right = np.full_like(lat_right, lon_max)

    lon_bottom = np.linspace(lon_max, lon_min, n_lon)
    lat_bottom = np.full_like(lon_bottom, lat_min)

    lat_left = np.linspace(lat_min, lat_max, n_lat)
    lon_left = np.full_like(lat_left, lon_min)

    poly_lon = np.concatenate([lon_top, lon_right, lon_bottom, lon_left])
    poly_lat = np.concatenate([lat_top, lat_right, lat_bottom, lat_left])
    verts = np.column_stack([poly_lon, poly_lat])
    codes = np.full(len(verts), Path.LINETO)
    codes[0] = Path.MOVETO
    return Path(verts, codes)


def set_globe_base(ax):
    """Set the global orthographic view and draw the land base layer."""
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND_FACE, edgecolor=COAST_COLOR, zorder=2)
    ax.coastlines(color=COAST_COLOR, zorder=3)


def draw_latitude_reference(ax):
    """Draw a dashed 34.5 S circle; Cartopy clips it to the visible globe."""
    lon_line = np.linspace(-180.0, 180.0, 1440)
    ax.plot(
        lon_line,
        np.full_like(lon_line, LAT_TARGET),
        color=LATITUDE_LINE_COLOR,
        linewidth=1.0,
        linestyle=(0, (8.0, 2.2)),
        transform=DATA_PROJECTION,
        zorder=0.8,
    )
    ax.text(
        -88.0,
        LAT_TARGET + 4.0,
        "34.5°S",
        transform=DATA_PROJECTION,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#666666",
        zorder=8,
    )


def draw_box_outline(ax, lon_min, lon_max, lat_min, lat_max, color, lw=2.0, z=6):
    """Draw the four edges of a lon/lat rectangle on a Cartopy axis."""
    lon_curve = np.linspace(lon_min, lon_max, 300)
    lat_curve = np.linspace(lat_min, lat_max, 160)

    ax.plot(lon_curve, np.full_like(lon_curve, lat_max), color=color, linewidth=lw,
            transform=DATA_PROJECTION, zorder=z)
    ax.plot(lon_curve, np.full_like(lon_curve, lat_min), color=color, linewidth=lw,
            transform=DATA_PROJECTION, zorder=z)
    ax.plot(np.full_like(lat_curve, lon_min), lat_curve, color=color, linewidth=lw,
            transform=DATA_PROJECTION, zorder=z)
    ax.plot(np.full_like(lat_curve, lon_max), lat_curve, color=color, linewidth=lw,
            transform=DATA_PROJECTION, zorder=z)


def curved_box(ax, lon_left, lon_right, lat_bottom, lat_top, facecolor,
               edgecolor=BIN_EDGE_COLOR, lw=2.0, n=180, z=6):
    """Fill a lon/lat rectangle as a curved polygon (edges follow great-circle linearity in Cartopy)."""
    lon_top = np.linspace(lon_left, lon_right, n)
    lat_top_arr = np.full_like(lon_top, lat_top)
    lon_bottom = np.linspace(lon_right, lon_left, n)
    lat_bottom_arr = np.full_like(lon_bottom, lat_bottom)

    ax.fill(
        np.concatenate([lon_top, lon_bottom]),
        np.concatenate([lat_top_arr, lat_bottom_arr]),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=DATA_PROJECTION,
        zorder=z,
    )




# --- Native field (left panel) and 20-bin averaged input (right panel) ---

native_values = np.asarray(sst_oras.values, dtype=float)
shared_norm = robust_norm(np.concatenate([native_values.ravel(), bin_means_sst]))  # shared colour range

sst_sm = mpl.cm.ScalarMappable(norm=shared_norm, cmap=SST_CMAP)
sst_sm.set_array([])

fig = plt.figure(figsize=(10.0, 6.05), constrained_layout=False)

# Two globe panels sitting side by side, with a shared horizontal colour bar below.
ax_native = fig.add_axes([-0.040, 0.170, 0.44, 0.760], projection=GLOBE_PROJECTION)
ax_bins = fig.add_axes([0.600, 0.170, 0.44, 0.760], projection=GLOBE_PROJECTION)

title_kwargs = dict(ha="center", va="top", fontsize=19.5, fontweight="normal", color="black")
for title_x, title_text in [
    (0.180, "(a) SST field in the reconstruction band"),
    (0.820, "(b) Binned input representation"),
]:
    fig.text(title_x, 0.985, title_text, **title_kwargs)

# Panel (a): native-grid globe. The pcolormesh is clipped to the lon/lat box so only
# the reconstruction band is coloured; the box outline is drawn on top for readability.
set_globe_base(ax_native)
draw_latitude_reference(ax_native)

pcm_native = ax_native.pcolormesh(
    lon2d.values,
    lat2d.values,
    native_values,
    transform=DATA_PROJECTION,
    cmap=SST_CMAP,
    norm=shared_norm,
    shading="auto",
    zorder=1,
)

native_box_path = make_box_path(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
native_box_patch = PathPatch(
    native_box_path,
    transform=DATA_PROJECTION._as_mpl_transform(ax_native),
    facecolor="none",
    edgecolor="none",
)
ax_native.add_patch(native_box_patch)
pcm_native.set_clip_path(native_box_patch)
draw_box_outline(ax_native, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, BOX_COLOR, lw=1.2)

# Panel (b): bin-averaged globe. Each bin is a coloured curved rectangle, grey for NaN.
set_globe_base(ax_bins)
draw_latitude_reference(ax_bins)

for i, value in enumerate(bin_means_sst):
    face = SST_CMAP(shared_norm(value)) if np.isfinite(value) else (0.85, 0.85, 0.85, 1.0)
    curved_box(
        ax_bins,
        bin_edges[i],
        bin_edges[i + 1],
        LAT_MIN,
        LAT_MAX,
        face,
        edgecolor=BIN_EDGE_COLOR,
        lw=0.9,
        z=6,
    )

# Shared colour bar centred under the two globe panels.
cax_shared = fig.add_axes([0.18, 0.065, 0.64, 0.035])
cb_shared = fig.colorbar(sst_sm, cax=cax_shared, orientation="horizontal")
cb_shared.set_label("Sea surface temperature [°C]", fontsize=13.7, labelpad=5)
cb_shared.ax.tick_params(labelsize=11.0, length=3.0, pad=4)

OUTPUT_PDF = OUTPUT_DIR / "fig03_sst_bins.pdf"
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(OUTPUT_PDF)
plt.close(fig)
