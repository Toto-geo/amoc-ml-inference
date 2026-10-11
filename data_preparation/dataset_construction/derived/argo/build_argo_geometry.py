#!/usr/bin/env python3
"""Rebuild the real Argo catalog and the virtual ORAS5 sampling geometry at 34.5 S."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr


GEOMETRY_NAME = "oras5_virtual_argo_coverage_geometry_34p5S_pm3deg_1958_2025.parquet"
LON_MIN, LON_MAX = -51.5, 17 + 33.46 / 60   # western and eastern longitudes of the sampling domain (deg)
LAT_MIN, LAT_MAX = -37.5, -31.5


def real_catalog(qc: pd.DataFrame) -> pd.DataFrame:
    """Select the 2012-2021 QC-filtered Argo profiles for the real catalog."""
    rows = qc.loc[(qc.date >= "2012-01-01") & (qc.date < "2022-01-01")].copy()
    rows = rows.sort_values("date")   # stable ordering at tied timestamps
    result = pd.DataFrame({
        "profile_id": rows.wmo.astype(str) + "_" + rows.cyc.map(lambda x: f"{x:04d}"),
        "wmo": rows.wmo,
        "cyc": rows.cyc,
        "time": rows.date,
        "lat": rows.lat,
        "lon": rows.lon,
    }).reset_index(drop=True)
    if result.profile_id.duplicated().any():
        raise ValueError("Duplicate Argo profile IDs")
    return result


def build(catalog: pd.DataFrame, temperature: xr.Dataset, mesh: xr.Dataset) -> pd.DataFrame:
    """Build the virtual Argo sampling geometry with jitter and bilinear-or-fallback stencils."""
    # T-grid geometry and the "eligible" ocean cells (wet surface and ocean depth >= 2000 m).
    longitude = temperature.nav_lon.values
    latitude = temperature.nav_lat.values
    if longitude.ndim == 3:
        longitude, latitude = longitude[0], latitude[0]
    depth = temperature.deptht.values
    y0, x0 = int(temperature.attrs["y_start_global"]), int(temperature.attrs["x_start_global"])
    y1, x1 = y0 + temperature.sizes["y"], x0 + temperature.sizes["x"]
    tmask = mesh.tmask.isel(t=0, y=slice(y0, y1), x=slice(x0, x1)).values
    wet_levels = tmask.sum(axis=0).astype(int)                     # wet levels per column
    deepest = np.full(wet_levels.shape, np.nan)
    wet = wet_levels > 0
    deepest[wet] = depth[wet_levels[wet] - 1]                      # deepest wet depth per column (m)
    eligible = (tmask[0] == 1) & (deepest >= 2000.0)               # surface wet and reaches >= 2000 m
    lon1d, lat1d = longitude[0].copy(), latitude[:, 0].copy()
    if (not np.allclose(longitude, lon1d[None, :])
            or not np.allclose(latitude, lat1d[:, None])):
        raise ValueError("The regional T grid is not separable into lon(x), lat(y)")

    # Template months (2013-2020): the real-Argo months replayed through 1958-2025.
    source = catalog.copy()
    source["time"] = pd.to_datetime(source.time)
    source["year_month"] = source.time.dt.to_period("M")
    source = source.loc[
        source.lon.between(LON_MIN, LON_MAX) & source.lat.between(LAT_MIN, LAT_MAX)
        & source.time.dt.year.between(2013, 2020)
    ]
    source = source[["profile_id", "wmo", "cyc", "time", "year_month", "lat", "lon"]]
    source = source.sort_values(["year_month", "time", "profile_id"]).copy()
    source["source_profile_rank_in_template_month"] = source.groupby("year_month").cumcount()
    source = source.rename(columns={"profile_id": "source_profile_id", "time": "source_time",
                                    "year_month": "template_month", "lat": "source_lat", "lon": "source_lon"})
    if source.template_month.nunique() != 96:
        raise ValueError("Missing 2013-2020 Argo template months")

    # Calendar: map every target month (1958-2025) to a template month and a cycle index.
    months = pd.period_range("1958-01", "2025-12", freq="M")
    templates = pd.period_range("2013-01", "2020-12", freq="M")
    calendar = pd.DataFrame({"target_month": months})
    calendar["cycle_id"] = np.arange(len(months)) // len(templates)    # number of wraps through the template set
    calendar["template_month"] = templates[np.arange(len(months)) % len(templates)]
    times = pd.DataFrame({"oras_time": pd.to_datetime(temperature.time_counter.values)})
    times["target_month"] = times.oras_time.dt.to_period("M")
    if len(times) != 816 or times.target_month.duplicated().any():
        raise ValueError("Expected one ORAS5 timestamp per month")
    frame = calendar.merge(source, on="template_month", how="left", validate="many_to_many")
    frame = frame.sort_values(["target_month", "source_profile_rank_in_template_month",
                               "source_time", "source_profile_id"]).copy()
    frame["virtual_profile_rank_in_month"] = frame.groupby("target_month").cumcount()
    frame["virtual_profile_id"] = np.arange(len(frame))
    frame = frame.merge(times, on="target_month", how="left", validate="many_to_one")

    # Jitter non-first-cycle samples by sigma=0.15 grid cells; retry until inside the band.
    lon0, lat0 = frame.source_lon.to_numpy(), frame.source_lat.to_numpy()
    cycle = frame.cycle_id.to_numpy()
    dx = np.diff(lon1d).mean()                                         # mean zonal grid spacing (deg)
    iy = np.clip(np.searchsorted(lat1d, lat0, side="left"), 1, len(lat1d) - 1)
    dy = (lat1d[iy] - lat1d[iy - 1])                                   # local meridional grid spacing (deg)
    lon, lat = lon0.copy(), lat0.copy()
    attempts = np.zeros(len(frame), dtype=int)
    pending = cycle > 0                                                # first cycle keeps the original position
    rng = np.random.default_rng(42)
    while pending.any():
        idx = np.where(pending)[0]
        eps_x = rng.normal(0.0, 0.15, len(idx))                        # Gaussian jitter in grid-cell units
        eps_y = rng.normal(0.0, 0.15, len(idx))
        candidate_lon = lon0[idx] + eps_x * dx
        candidate_lat = lat0[idx] + eps_y * dy[idx]
        accept = ((candidate_lon >= LON_MIN) & (candidate_lon <= LON_MAX)
                  & (candidate_lat >= LAT_MIN) & (candidate_lat <= LAT_MAX))
        lon[idx[accept]], lat[idx[accept]] = candidate_lon[accept], candidate_lat[accept]
        attempts[idx] += 1
        pending[idx] = ~accept

    # Bilinear-stencil test: the four surrounding cells must be wet and reach >= 2000 m.
    ix1 = np.searchsorted(lon1d, lon, side="right")
    iy1 = np.searchsorted(lat1d, lat, side="right")
    ix0, iy0 = ix1 - 1, iy1 - 1                                        # lower-left corner of the stencil
    inside = (ix0 >= 0) & (ix1 < len(lon1d)) & (iy0 >= 0) & (iy1 < len(lat1d))
    bilinear = np.zeros(len(frame), dtype=bool)
    all_wet = np.zeros(len(frame), dtype=bool)
    for index in np.where(inside)[0]:
        wet_block = tmask[0, iy0[index]:iy1[index] + 1, ix0[index]:ix1[index] + 1]
        valid_block = eligible[iy0[index]:iy1[index] + 1, ix0[index]:ix1[index] + 1]
        all_wet[index] = np.all(wet_block == 1)
        bilinear[index] = all_wet[index] and np.all(valid_block)
    fallback = ~bilinear
    reason = np.full(len(frame), "bilinear_ok", dtype=object)
    reason[~inside] = "outside_bilinear_bbox"
    reason[inside & ~all_wet] = "stencil_not_all_wet"
    reason[inside & all_wet & ~bilinear] = "stencil_not_all_valid_2000"

    # Nearest-eligible fallback for samples that fail the bilinear test.
    fy, fx = np.where(eligible)
    nearest = cKDTree(np.column_stack([longitude[eligible], latitude[eligible]]))
    fb_idx = np.where(fallback)[0]
    _, nearest_idx = nearest.query(np.column_stack([lon[fb_idx], lat[fb_idx]]), k=1)
    ref_y, ref_x = iy0.copy(), ix0.copy()
    ref_y[fb_idx], ref_x[fb_idx] = fy[nearest_idx], fx[nearest_idx]
    ref_lat, ref_lon = latitude[ref_y, ref_x], longitude[ref_y, ref_x]
    fb_dist = np.full(len(frame), np.nan)
    fb_dx = (lon[fb_idx] - ref_lon[fb_idx]) * 111.32 * np.cos(
        np.deg2rad((lat[fb_idx] + ref_lat[fb_idx]) / 2.0))
    fb_dy = (lat[fb_idx] - ref_lat[fb_idx]) * 111.32
    fb_dist[fb_idx] = np.sqrt(fb_dx**2 + fb_dy**2)                     # great-circle fallback distance (km)

    # Bilinear weights where the stencil is valid; NaN elsewhere.
    tx = np.full(len(frame), np.nan)
    ty = tx.copy()
    ok = np.where(bilinear)[0]
    tx[ok] = (lon[ok] - lon1d[ix0[ok]]) / (lon1d[ix1[ok]] - lon1d[ix0[ok]])
    ty[ok] = (lat[ok] - lat1d[iy0[ok]]) / (lat1d[iy1[ok]] - lat1d[iy0[ok]])
    w00, w10 = (1 - tx) * (1 - ty), tx * (1 - ty)
    w01, w11 = (1 - tx) * ty, tx * ty
    fx0, fx1, fy0, fy1 = (pd.Series(np.where(bilinear, x, pd.NA), dtype="Int64")
                            for x in (ix0, ix1, iy0, iy1))

    # Jitter diagnostics in grid-cell units and great-circle kilometres.
    jitter_x, jitter_y = (lon - lon0) / dx, (lat - lat0) / dy
    jitter_km_x = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    jitter_km_y = (lat - lat0) * 111.32

    # Assemble the long-format output row for every virtual profile.
    result = frame[["virtual_profile_id", "virtual_profile_rank_in_month", "target_month",
                    "oras_time", "cycle_id", "template_month", "source_profile_id", "wmo",
                    "cyc", "source_time", "source_profile_rank_in_template_month",
                    "source_lat", "source_lon"]].copy()
    result["jitter_applied"] = cycle > 0
    result["jitter_attempts"] = attempts
    result["jitter_norm_cells"] = np.sqrt(jitter_x**2 + jitter_y**2)
    result["jitter_norm_km"] = np.sqrt(jitter_km_x**2 + jitter_km_y**2)
    result["sample_lat"], result["sample_lon"] = lat, lon
    result["sampling_mode"] = np.where(bilinear, "bilinear", "fallback_nearest")
    result["bilinear_ok"], result["need_fallback"] = bilinear, fallback
    result["fallback_reason"] = reason
    result["fallback_dist_km"] = fb_dist
    result["ref_t_y"] = pd.Series(ref_y, dtype="Int64")
    result["ref_t_x"] = pd.Series(ref_x, dtype="Int64")
    result["ref_t_lat"], result["ref_t_lon"] = ref_lat.astype("f8"), ref_lon.astype("f8")
    for name, values in (("bilin_ix0", fx0), ("bilin_ix1", fx1),
                         ("bilin_iy0", fy0), ("bilin_iy1", fy1),
                         ("bilin_w00", w00), ("bilin_w10", w10),
                         ("bilin_w01", w01), ("bilin_w11", w11)):
        result[name] = values
    result["bilin_weight_sum"] = result[["bilin_w00", "bilin_w10", "bilin_w01", "bilin_w11"]].fillna(0.0).sum(axis=1)
    result["target_month"] = result.target_month.astype(str)
    result["template_month"] = result.template_month.astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-csv", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--mesh-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog_path = args.output_dir / "03_argo_real_coverage.csv"
    geometry_path = args.output_dir / GEOMETRY_NAME
    if catalog_path.exists() or geometry_path.exists():
        raise FileExistsError("Argo catalog or geometry already exists")
    catalog = real_catalog(pd.read_csv(args.qc_csv))
    with xr.open_dataset(args.temperature) as temperature, xr.open_dataset(args.mesh_mask) as mesh:
        geometry = build(catalog, temperature, mesh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    geometry.to_parquet(geometry_path, index=False)
    print(f"Wrote {len(catalog)} real and {len(geometry)} virtual Argo profiles")


if __name__ == "__main__":
    main()
