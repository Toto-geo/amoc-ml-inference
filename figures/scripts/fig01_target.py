#!/usr/bin/env python
# coding: utf-8
"""Figure 1: AMOC target section, mean overturning profile and monthly transport.

Three panels: (a) the A-Z target section at 34.5 S on an orthographic globe,
(b) the full-period mean overturning streamfunction with the selected target
depth, and (c) the monthly target time series used by the model.
"""

# --- Repo bootstrap and runtime cache ---

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data_preparation/data"
ORAS5_DYNAMIC_DIR = DATA / "dynamic"
ORAS5_TARGET_DIR = DATA / "target"
parser = argparse.ArgumentParser(description="Recreate paper Figure 1: AMOC target")
parser.add_argument("--output-dir", type=Path, required=True)
OUTPUT_DIR = parser.parse_args().output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Load the monthly target and reconstruct the matching streamfunction profile ---

import numpy as np
import pandas as pd
import xarray as xr

TARGET_MONTHLY_NC = ORAS5_TARGET_DIR / "monthly" / "oras5_monthly_target_34p5S_AZ_medianDepth_1958_2025.nc"
VOMECRTY_SECTION_NC = ORAS5_DYNAMIC_DIR / "meridional_velocity" / "oras5_vomecrty_34p5S_1958_2025.nc"
EKMAN_VELOCITY_NC = ORAS5_DYNAMIC_DIR / "ekman_velocity" / "oras5_ekman_velocity_34p5S_1958_2025.nc"

for label, path in {
    "monthly target": TARGET_MONTHLY_NC,
    "vomecrty section": VOMECRTY_SECTION_NC,
    "ekman velocity": EKMAN_VELOCITY_NC,
}.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

# Monthly M-regime target used by the model.
ds_target = xr.open_dataset(TARGET_MONTHLY_NC)
ds_v = xr.open_dataset(VOMECRTY_SECTION_NC)
ds_e = xr.open_dataset(EKMAN_VELOCITY_NC)

time = pd.DatetimeIndex(ds_target["time_counter"].values)
target_monthly = np.asarray(ds_target["target_monthly"].values, dtype=float)

i_A = int(ds_target.attrs["A_index_on_V_grid"])
i_Z = int(ds_target.attrs["Z_index_on_V_grid"])
k_target = int(ds_target["k_target"].values)
depth_target = float(ds_target["depth_target"].values)


lon_v = np.asarray(ds_e["lon_v"].values, dtype=float)
lat_v = np.asarray(ds_e["lat_v"].values, dtype=float)
depth_v = np.asarray(ds_v["depthv"].values, dtype=float)

# A-Z section geometry on the V grid.
section_slice = slice(i_A, i_Z + 1)
section_lon = lon_v[section_slice]
section_lat = lat_v[section_slice]
A_lon, A_lat = float(lon_v[i_A]), float(lat_v[i_A])
Z_lon, Z_lat = float(lon_v[i_Z]), float(lat_v[i_Z])

# Full Atlantic wet section; used in panel (a) only to show the inshore segments.
atlantic_wet_idx = np.flatnonzero(np.asarray(ds_e["m_Atl_V"].values).astype(bool))
atl_west_idx = int(atlantic_wet_idx[0])
atl_east_idx = int(atlantic_wet_idx[-1])
west_inshore_lon = lon_v[atl_west_idx : i_A + 1]
west_inshore_lat = lat_v[atl_west_idx : i_A + 1]
east_inshore_lon = lon_v[i_Z : atl_east_idx + 1]
east_inshore_lat = lat_v[i_Z : atl_east_idx + 1]

# Reconstruct Psi_AZ via the full target-construction sequence:
# raw vomecrty -> full-section barotropic mass-balance correction -> Ekman removal -> A-Z restriction.
v_tot = np.asarray(ds_v["vomecrty"].isel(y=0).values, dtype=np.float32)       # (time, depthv, x)
v_ek = np.asarray(ds_e["v_ek_eq"].values, dtype=np.float32)                   # (time, depthv, x)
dz_V = np.asarray(ds_e["dz_V"].values, dtype=np.float64)                      # (depthv, x)
e1v = np.asarray(ds_e["e1v"].values, dtype=np.float64)                        # (x,)
m_Atl_V = np.asarray(ds_e["m_Atl_V"].values).astype(bool)                     # (x,)

if v_tot.shape != v_ek.shape:
    raise ValueError(f"v_tot and v_ek shape mismatch: {v_tot.shape} != {v_ek.shape}")
if not np.array_equal(ds_target["time_counter"].values, ds_v["time_counter"].values):
    raise ValueError("Target and vomecrty time axes are not identical.")
if not np.array_equal(ds_target["time_counter"].values, ds_e["time_counter"].values):
    raise ValueError("Target and Ekman time axes are not identical.")

A_V = e1v[None, :] * dz_V                                                      # (depthv, x) cell area
m_Atl_2d = np.broadcast_to(m_Atl_V[None, :], A_V.shape)
A_V_atl = np.where(m_Atl_2d, A_V, 0.0)

q_raw_full = np.where(m_Atl_2d[None, :, :], v_tot * A_V[None, :, :], 0.0)
T_raw_full = np.nansum(q_raw_full, axis=(1, 2))
A_tot_full = float(np.nansum(A_V_atl))
v0_full = T_raw_full / A_tot_full                                              # section-mean v per month (barotropic drift)

A_AZ = A_V[:, section_slice]
v_corr_AZ = v_tot[:, :, section_slice] - v0_full[:, None, None] - v_ek[:, :, section_slice]
q_corr_AZ = v_corr_AZ * A_AZ[None, :, :]
Q_AZ = np.nansum(q_corr_AZ, axis=2)                                            # zonal integral over A-Z
Psi_AZ = np.cumsum(Q_AZ, axis=1)                                               # surface-to-bottom cumulative sum
psi_profile_mean = np.mean(Psi_AZ / 1e6, axis=0)                               # m3/s to Sv, time mean

# Cross-check: the reconstructed depth-k_target series must match the saved monthly target.
reconstructed_target = Psi_AZ[:, k_target] / 1e6
max_abs_target_diff = float(np.nanmax(np.abs(reconstructed_target - target_monthly)))
if max_abs_target_diff > 5e-4:
    raise ValueError(
        "Reconstructed target does not match saved monthly target closely enough: "
        f"max_abs_diff={max_abs_target_diff:.6e} Sv"
    )

print("=" * 100)
print("TARGET DATA USED")
print("=" * 100)
print(f"Target file       : {TARGET_MONTHLY_NC}")
print(f"Target variable   : target_monthly")
print(f"Time range        : {time[0].strftime('%Y-%m')} to {time[-1].strftime('%Y-%m')} ({len(time)} months)")
print(f"A index/lon/lat   : {i_A} | {A_lon:.6f} | {A_lat:.6f}")
print(f"Z index/lon/lat   : {i_Z} | {Z_lon:.6f} | {Z_lat:.6f}")
print(f"Target depth      : k={k_target}, depth={depth_target:.6f} m")
print(f"Target range      : {np.nanmin(target_monthly):.3f} to {np.nanmax(target_monthly):.3f} Sv")
print(f"Profile range     : {np.nanmin(psi_profile_mean):.3f} to {np.nanmax(psi_profile_mean):.3f} Sv")
print(f"Max reconstruction difference at target depth: {max_abs_target_diff:.3e} Sv")




# --- Plot helpers and style ---

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

GLOBE_PROJECTION = ccrs.Orthographic(central_longitude=-17.0, central_latitude=-34.5)
DATA_PROJECTION = ccrs.PlateCarree()
SOUTH_ATLANTIC_EXTENT = [-58.0, 24.0, -63.0, -4.0]      # lon_min, lon_max, lat_min, lat_max

LAND_FACE = "#ece8d5"
COAST_COLOR = "0.35"
ACCENT_BLUE = "#2d9cdb"
ACCENT_BLUE_DARK = "#1f78b4"
LATITUDE_LINE_COLOR = "#666666"
INSHORE_SEGMENT_COLOR = "#d28a5a"
GRID_COLOR = "0.88"
ARROW_COLOR = "0.25"
TARGET_LINE_COLOR = "0.35"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 19.5,
    "xtick.labelsize": 16.2,
    "ytick.labelsize": 16.2,
    "legend.fontsize": 18.4,
})

LAT_TARGET = -34.5


def set_globe_base(ax):
    """Set the South Atlantic extent and draw the land/ocean base layers."""
    ax.set_extent(SOUTH_ATLANTIC_EXTENT, crs=DATA_PROJECTION)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_FACE, edgecolor=COAST_COLOR, linewidth=0.5, zorder=1)
    ax.coastlines(color=COAST_COLOR, linewidth=0.5, zorder=2)


def draw_latitude_reference(ax, label_lon=60.0, label_lat=None):
    """Draw a dashed 34.5 S reference line with a text label."""
    if label_lat is None:
        label_lat = LAT_TARGET - 11.5
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
        label_lon,
        label_lat,
        "34.5°S",
        transform=DATA_PROJECTION,
        ha="center",
        va="center",
        fontsize=15.0,
        color=LATITUDE_LINE_COLOR,
        zorder=8,
    )


def clean_spines(ax):
    """Hide top/right spines and soften the remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("0.35")
        ax.spines[spine].set_linewidth(0.8)


def add_endpoint_label(ax, lon, lat, label, dx, dy):
    """Place a bold text label with a white outline near a map endpoint."""
    text = ax.text(
        lon + dx,
        lat + dy,
        label,
        transform=DATA_PROJECTION,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        zorder=10,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
    return text




# --- Composite target figure: section -> streamfunction profile -> time series ---

fig = plt.figure(figsize=(22, 6.8), dpi=150, constrained_layout=False)

# Three fixed axes in figure coordinates: map, profile, time series.
ax_map = fig.add_axes([0.0425, 0.190, 0.260, 0.640], projection=GLOBE_PROJECTION)
ax_profile = fig.add_axes([0.363, 0.190, 0.145, 0.640])
ax_series = fig.add_axes([0.575, 0.285, 0.395, 0.450])

title_kwargs = dict(ha="center", va="top", fontsize=23.9, fontweight="normal", color="black")
for title_x, title_text in [
    (0.1725, "(a) 34.5°S target section"),
    (0.4355, "(b) Mean overturning streamfunction"),
    (0.7725, "(c) Target transport time series"),
]:
    fig.text(title_x, 0.955, title_text, **title_kwargs)

# Panel (a): globe with the A-Z target section.
set_globe_base(ax_map)
draw_latitude_reference(
    ax_map,
    label_lon=float(0.5 * (A_lon + Z_lon)),
    label_lat=float(np.nanmean(section_lat) + 4.0),
)

# Draw inshore segments underneath so the A-Z line (zorder 6) sits on top.
ax_map.plot(
    west_inshore_lon,
    west_inshore_lat,
    transform=DATA_PROJECTION,
    color=INSHORE_SEGMENT_COLOR,
    linewidth=2.8,
    solid_capstyle="round",
    zorder=5.8,
)
ax_map.plot(
    section_lon,
    section_lat,
    transform=DATA_PROJECTION,
    color=ACCENT_BLUE_DARK,
    linewidth=2.8,
    solid_capstyle="round",
    zorder=6,
)
ax_map.plot(
    east_inshore_lon,
    east_inshore_lat,
    transform=DATA_PROJECTION,
    color=INSHORE_SEGMENT_COLOR,
    linewidth=2.8,
    solid_capstyle="round",
    zorder=5.8,
)

map_legend_handles = [
    Line2D([0], [0], color=ACCENT_BLUE_DARK, linewidth=2.8, solid_capstyle="round"),
    Line2D([0], [0], color=INSHORE_SEGMENT_COLOR, linewidth=2.8, solid_capstyle="round"),
]
# Anchor the legend below the map, centred on its horizontal axis.
map_legend_anchor = (
    ax_map.get_position().x0 + 0.5 * ax_map.get_position().width,
    ax_profile.get_position().y0 - 0.090,
)
fig.legend(
    map_legend_handles,
    ["SAMBA interior section", "Inshore segments"],
    loc="center",
    bbox_to_anchor=map_legend_anchor,
    ncol=2,
    frameon=False,
    handlelength=2.6,
    handletextpad=0.6,
    columnspacing=1.6,
    borderaxespad=0.0,
)

# Panel (b): full-period mean overturning streamfunction with a dashed target depth.
ax_profile.plot(
    psi_profile_mean,
    depth_v,
    color=ACCENT_BLUE_DARK,
    linewidth=2.2,
    label="_nolegend_",
)
ax_profile.axhline(
    depth_target,
    color=TARGET_LINE_COLOR,
    linewidth=1.2,
    linestyle=(0, (5.0, 2.2)),
    label="Target depth",
)
ax_profile.invert_yaxis()                              # depth increases downward
ax_profile.set_xlabel("Transport [Sv]", labelpad=9)
ax_profile.set_ylabel("Depth [m]", labelpad=16)
ax_profile.yaxis.set_label_coords(-0.255, 0.5)
ax_profile.grid(True, color=GRID_COLOR, linewidth=0.6)
clean_spines(ax_profile)
ax_profile.tick_params(axis="both", length=3.5, pad=5)
ax_profile.legend(
    loc="lower right",
    frameon=False,
    handlelength=2.4,
    fontsize=18.4,
    borderaxespad=0.0,
)

profile_pad = 0.08 * (np.nanmax(psi_profile_mean) - np.nanmin(psi_profile_mean))
ax_profile.set_xlim(np.nanmin(psi_profile_mean) - profile_pad, np.nanmax(psi_profile_mean) + profile_pad)
ax_profile.set_ylim(depth_v[-1], 0)

# Panel (c): monthly target transport at the selected depth.
ax_series.plot(
    time,
    target_monthly,
    color=ACCENT_BLUE_DARK,
    linewidth=1.15,
)
ax_series.set_xlabel("Year", labelpad=9)
ax_series.set_ylabel("Transport at target depth [Sv]", labelpad=10)
ax_series.grid(True, color=GRID_COLOR, linewidth=0.6)
clean_spines(ax_series)
ax_series.tick_params(axis="both", length=3.5, pad=5)
ax_series.xaxis.set_major_locator(mdates.YearLocator(10))
ax_series.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_series.set_xlim(time[0], time[-1])
y_pad = 0.08 * (np.nanmax(target_monthly) - np.nanmin(target_monthly))
ax_series.set_ylim(np.nanmin(target_monthly) - y_pad, np.nanmax(target_monthly) + y_pad)


OUTPUT_PDF = OUTPUT_DIR / "fig01_target.pdf"
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(OUTPUT_PDF)
plt.close(fig)
