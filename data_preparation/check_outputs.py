#!/usr/bin/env python3
"""Check the final data contract after running data preparation."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# netCDF4's wheel is compiled against a slightly older NumPy C-ABI than the pinned
# runtime, which emits a harmless "numpy.ndarray size changed" RuntimeWarning on first
# import. Data and results are unaffected; silence only this specific message.
warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)


DEFAULT_DATA = Path(__file__).resolve().parent / "data"
SCALES = ((20, 20), (40, 10), (80, 5), (160, 3))   # (n_bins, n_lat_rows)
FIELDS = {
    "SST": ("sst", "sst_bin_means"),
    "SSS": ("sss", "sss_bin_means"),
    "SSH": ("ssh", "ssh_bin_means"),
    "OBP": ("virtual_wlt", "obp_bin_means"),
}
FIGURE_FIELDS = {
    "dynamic/SST/oras5_sosstsst_34p5S_pm2deg_1958_2025.nc": ("sosstsst", (816, 20, 295)),
    "dynamic/Temperature/oras5_votemper_34p5S_pm3deg_1958_2025.nc": ("votemper", (816, 75, 30, 295)),
    "dynamic/meridional_velocity/oras5_vomecrty_34p5S_1958_2025.nc": ("vomecrty", (816, 75, 1, 295)),
    "dynamic/ekman_velocity/oras5_ekman_velocity_34p5S_1958_2025.nc": ("v_ek_eq", (816, 75, 295)),
}


def require(condition: bool, path: Path, detail: str) -> None:
    if not condition:
        raise ValueError(f"{path}: {detail}")


def check(data: Path) -> None:
    """Validate every product under data: targets, Argo inputs, surface inputs, figure fields."""
    if not data.exists():
        raise SystemExit(
            f"Prepared data not found at '{data}'. Extract data.zip from the Zenodo record "
            "(https://doi.org/10.5281/zenodo.23287700) into the repository root, or rebuild it "
            "following data_preparation/README.md, before running this check."
        )
    # Monthly and quarterly AMOC targets.
    monthly = data / "target/monthly/oras5_monthly_target_34p5S_AZ_medianDepth_1958_2025.nc"
    quarterly = data / "target/quarterly/oras5_quarterly_target_34p5S_AZ_medianDepth_1958_2025.nc"
    with xr.open_dataset(monthly) as ds:
        time = ds.time_counter.values              # reference monthly axis reused by later checks
        require(time.shape == (816,) and bool(np.isfinite(ds.target_monthly.values).all()),
                monthly, "monthly target must have 816 finite values")
    with xr.open_dataset(quarterly) as ds:
        require(ds.time_counter.size == 272 and bool(np.isfinite(ds.target_quarterly.values).all()),
                quarterly, "quarterly target must have 272 finite values")
    # Virtual-Argo density inputs (monthly and quarterly).
    for suffix, dim, size in (("", "time", 816), ("Q_", "sample", 272)):
        path = data / "input/virtual_argo" / f"oras5_virtual_argo_density_input_{suffix}34p5S_pm3deg_1958_2025.nc"
        with xr.open_dataset(path) as ds:
            require(ds.sizes[dim] == size and "rho" in ds and "profile_mask" in ds,
                    path, "unexpected Argo dimensions or variables")
            require(int(ds.profile_mask.values.sum()) > 0, path, "no virtual profiles")
            if dim == "time":
                require(bool(np.array_equal(ds.time.values, time)), path, "monthly time mismatch")
    # Four surface fields at four A-Z scales.
    for folder, (prefix, variable) in FIELDS.items():
        for bins, rows in SCALES:
            suffix = "" if rows == 20 else f"_lat{rows}"
            path = data / "input" / folder / f"oras5_{prefix}_input_AZ_{bins}bin{suffix}_means_1958_2025.nc"
            with xr.open_dataset(path) as ds:
                values = ds[variable].values
                require(values.shape == (816, bins) and bool(np.isfinite(values).all()),
                        path, "surface shape or finite-value check failed")
                require(bool(np.array_equal(ds.time_counter.values, time)), path, "monthly time mismatch")
                require(int(ds.attrs["surface_lat_rows"]) == rows, path, "latitude-row mismatch")
                require(bool((ds.bin_n_ocean_cells.values > 0).all()), path, "empty ocean bin")
    # Figure-only inputs (not consumed by the ML pipeline).
    for relative, (variable, shape) in FIGURE_FIELDS.items():
        path = data / relative
        with xr.open_dataset(path) as ds:
            require(variable in ds and ds[variable].shape == shape, path, "figure field shape mismatch")
            require(bool(np.array_equal(ds.time_counter.values, time)), path, "figure field time mismatch")
    # Real Argo catalog used by Figure 2.
    catalog = data / "argo/03_argo_real_coverage.csv"
    real = pd.read_csv(catalog, usecols=["time", "lon", "lat"], parse_dates=["time"])
    require(not real.empty and bool(real.time.notna().all())
            and bool(np.isfinite(real[["lon", "lat"]].to_numpy()).all()),
            catalog, "real Argo catalog has invalid time or coordinates")
    require(bool((real.time.dt.to_period("M") == pd.Period("2013-01")).any()),
            catalog, "figure-2 reference month is missing")
    print("Final data contract: 20 ML products and 5 figure inputs valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    check(args.data_root)


if __name__ == "__main__":
    main()
