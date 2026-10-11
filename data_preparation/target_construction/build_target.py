#!/usr/bin/env python3
"""Build the monthly and quarterly ORAS5 AMOC targets at 34.5 S."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


MONTHLY = "oras5_monthly_target_34p5S_AZ_medianDepth_1958_2025.nc"
QUARTERLY = "oras5_quarterly_target_34p5S_AZ_medianDepth_1958_2025.nc"
A_INDEX, Z_INDEX = 10, 286        # inclusive west/east indices of the Atlantic section on the V-grid
UPPER_CELL_MAX_DEPTH_M = 2000.0   # depth cap for the AMOC peak search


def build(velocity: xr.Dataset, ekman: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute the monthly and quarterly AMOC targets in Sv."""
    # Check input shapes and time axis.
    time = velocity.time_counter.values
    expected = np.arange(np.datetime64("1958-01"), np.datetime64("2026-01"), dtype="datetime64[M]")
    if (velocity.sizes.get("y") != 1 or velocity.sizes.get("x") != 295
            or velocity.sizes.get("depthv") != 75
            or not np.array_equal(time.astype("datetime64[M]"), expected)
            or not np.array_equal(time, ekman.time_counter.values)
            or not np.array_equal(velocity.depthv.values, ekman.depthv.values)):
        raise ValueError("Velocity and Ekman must describe the same 1958-2025 V section")

    lon = ekman.lon_v.values
    atlantic = ekman.m_Atl_V.values == 1   # Atlantic cells on the V-grid
    if (ekman.sizes.get("x") != 295 or ekman.sizes.get("depthv") != 75
            or not np.all(atlantic[A_INDEX:Z_INDEX + 1])
            or float(lon[A_INDEX]) != -51.5 or float(lon[Z_INDEX]) != 17.5):
        raise ValueError("A-Z geometry differs from the canonical V-grid section")

    # Section geometry and streamfunction container.
    area = ekman.e1v.values[None, :] * ekman.dz_V.values          # (depth, x) cell area in m^2
    atlantic_2d = np.broadcast_to(atlantic[None, :], area.shape)  # Atlantic mask aligned to area
    total_area = np.nansum(np.where(atlantic_2d, area, 0.0))      # full A-Z section area in m^2
    depth = velocity.depthv.values                                # 75 ORAS5 depth levels
    psi = np.empty((len(time), len(depth)), dtype=np.float64)     # streamfunction (time, depth)

    # Build the streamfunction one year at a time to bound memory.
    for start in range(0, len(time), 12):
        stop = min(start + 12, len(time))
        v = velocity.vomecrty.isel(time_counter=slice(start, stop), y=0).values    # (t, depth, x) m/s
        v_ek = ekman.v_ek_eq.isel(time_counter=slice(start, stop)).values          # (t, depth) m/s
        raw = np.where(atlantic_2d[None, :, :], v * area[None, :, :], 0.0)
        barotropic = np.nansum(raw, axis=(1, 2)) / total_area                      # section-mean v per month
        corrected = (v - barotropic[:, None, None]) - v_ek
        transport = corrected[:, :, A_INDEX:Z_INDEX + 1] * area[None, :, A_INDEX:Z_INDEX + 1]
        psi[start:stop] = np.cumsum(np.nansum(transport, axis=2), axis=1)          # zonal sum then cumsum in depth

    # Select the target depth and read the monthly series at that level.
    upper = np.flatnonzero(depth <= UPPER_CELL_MAX_DEPTH_M)                        # indices above 2000 m
    peak_depth = depth[upper[np.argmax(psi[:, upper], axis=1)]].astype(np.float64) # per-month peak depth
    median_depth = float(np.median(peak_depth))                                    # median across months
    target_index = int(np.argmin(np.abs(depth - median_depth)))                    # nearest ORAS5 level
    monthly_value = (psi[:, target_index] / 1e6).astype(np.float64)                # m^3/s to Sv
    quarterly_value = monthly_value.reshape(-1, 3).mean(axis=1)                    # JFM, AMJ, JAS, OND means
    quarter_time = time.reshape(-1, 3)[:, 2]                                       # timestamp at third month

    # Metadata shared by the two output datasets.
    common = {
        "method_depth_selection": "Monthly maximum of surface-to-bottom A-Z streamfunction above 2000 m; median peak depth, nearest ORAS5 level",
        "UPPER_CELL_MAX_DEPTH_M": UPPER_CELL_MAX_DEPTH_M,
        "A_index_on_V_grid": A_INDEX,
        "Z_index_on_V_grid": Z_INDEX,
        "A_longitude_on_V_grid_deg": float(lon[A_INDEX]),
        "Z_longitude_on_V_grid_deg": float(lon[Z_INDEX]),
        "velocity_correction_sequence": "Raw vomecrty -> full-section barotropic mass-balance correction -> Ekman removal -> A-Z restriction",
        "streamfunction_definition": "Surface-to-bottom cumulative sum of zonally integrated transport on corrected A-Z section",
    }
    depth_vars = {
        "k_target": ((), np.int32(target_index), {"long_name": "Selected depth index"}),
        "depth_target": ((), np.float32(depth[target_index]), {"long_name": "Selected ORAS5 depth", "units": "m"}),
        "z_star_median": ((), np.float32(median_depth), {"long_name": "Median monthly maximum depth", "units": "m"}),
    }
    monthly = xr.Dataset(
        data_vars={"target_monthly": (("time_counter",), monthly_value.astype("f4"),
                                      {"long_name": "Monthly AMOC at selected depth", "units": "Sv"}),
                   **depth_vars},
        coords={"time_counter": ("time_counter", time)},
        attrs={**common, "description": "ORAS5 34.5S A-Z monthly AMOC target, median-depth method"},
    )
    quarterly = xr.Dataset(
        data_vars={"target_quarterly": (("time_counter",), quarterly_value.astype("f4"),
                                        {"long_name": "Mean of three consecutive monthly AMOC targets", "units": "Sv"}),
                   **depth_vars},
        coords={"time_counter": ("time_counter", quarter_time)},
        attrs={**common, "description": "ORAS5 34.5S A-Z quarterly AMOC target, median-depth method",
               "quarterly_aggregation_definition": "Non-overlapping JFM, AMJ, JAS, OND means; timestamp at third month"},
    )
    return monthly, quarterly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--ekman", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_dir / "monthly" / MONTHLY,
               args.output_dir / "quarterly" / QUARTERLY)
    for path in outputs:
        if path.exists() or path.with_name(path.name + ".partial").exists():
            raise FileExistsError(path)
    with xr.open_dataset(args.velocity) as velocity, xr.open_dataset(args.ekman) as ekman:
        products = build(velocity, ekman)
    for product, path in zip(products, outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")   # staging file
        try:
            product.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(path)                          # rename when fully written
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Wrote {path}: {dict(product.sizes)}", flush=True)


if __name__ == "__main__":
    main()
