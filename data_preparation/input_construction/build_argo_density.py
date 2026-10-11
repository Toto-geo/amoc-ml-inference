#!/usr/bin/env python3
"""Build the monthly and calendar-quarter virtual-Argo density inputs.

Each virtual profile is sampled from the ORAS5 regional T/S section using
either a bilinear or nearest-eligible stencil, masked above its assigned
QC pressure, and summarised as in-situ density in 20 m depth bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr


MONTHLY = "oras5_virtual_argo_density_input_34p5S_pm3deg_1958_2025.nc"
QUARTERLY = "oras5_virtual_argo_density_input_Q_34p5S_pm3deg_1958_2025.nc"
PHYSICAL = ("temperature", "salinity", "SA", "CT", "rho")
VARIABLE_ATTRS = {
    "temperature": {"units": "degC", "long_name": "in situ temperature sampled from ORAS5"},
    "salinity": {"units": "PSU", "long_name": "practical salinity sampled from ORAS5"},
    "SA": {"units": "g kg-1", "long_name": "absolute salinity"},
    "CT": {"units": "degC", "long_name": "conservative temperature"},
    "rho": {"units": "kg m-3", "long_name": "in situ density after 20 m smoothing"},
    "lon": {"units": "degrees_east"},
    "lat": {"units": "degrees_north"},
    "z": {"units": "m", "long_name": "native ORAS5 depth levels"},
    "pressure": {"units": "dbar", "long_name": "common reference pressure axis"},
    "month_frac": {"units": "1", "long_name": "relative time within monthly window"},
    "month_in_quarter": {"long_name": "month index inside calendar quarter", "values": "0,1,2"},
}


def annotate(dataset: xr.Dataset) -> xr.Dataset:
    """Attach the shared variable attributes to every known field in dataset."""
    for name, attrs in VARIABLE_ATTRS.items():
        if name in dataset:
            dataset[name].attrs = attrs
    return dataset


def assigned_pressure(geometry: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """Attach each virtual profile a QC pressure limit, shuffling later cycles within their template month."""
    rows = geometry.merge(qc[["wmo", "cyc", "max_pres_good"]], on=["wmo", "cyc"],
                          how="left", validate="many_to_one")
    if rows.max_pres_good.isna().any() or len(rows) != len(geometry):
        raise ValueError("Geometry profiles must all have a QC pressure limit")
    rows["assigned_max_pres_good"] = np.nan
    # First cycle keeps its original QC pressure; later cycles get a permutation
    # of the same template-month pool, so every month reuses the same QC values.
    first_cycle = rows.cycle_id == 0
    rows.loc[first_cycle, "assigned_max_pres_good"] = rows.loc[first_cycle, "max_pres_good"].to_numpy()
    pools = rows.loc[first_cycle, ["template_month", "max_pres_good"]]
    rng = np.random.default_rng(123)
    for month, pool in pools.groupby("template_month", sort=True):
        values = pool.max_pres_good.to_numpy().copy()
        indexes = rows.index[(~first_cycle) & (rows.template_month == month)].to_numpy()
        for start in range(0, len(indexes), len(values)):
            block = indexes[start:start + len(values)]
            rows.loc[block, "assigned_max_pres_good"] = rng.permutation(values)[:len(block)]
    if rows.assigned_max_pres_good.isna().any():
        raise ValueError("Incomplete pressure assignment")
    return rows


def monthly_input(rows: pd.DataFrame, temperature: xr.Dataset, salinity: xr.Dataset) -> xr.Dataset:
    """Sample ORAS5 T/S for every virtual profile and compute binned in-situ density."""
    if not np.array_equal(temperature.time_counter.values, salinity.time_counter.values):
        raise ValueError("Temperature and salinity time axes differ")
    depth = temperature.deptht.values.astype(float)
    depth = depth[depth <= 2000.0]                                               # keep only 0-2000 dbar levels
    n_time = temperature.sizes["time_counter"]
    n_pos = int(rows.groupby("target_month").virtual_profile_rank_in_month.max().max() + 1)
    n_z = len(depth)
    shape3 = (n_time, n_pos, n_z)
    shape2 = (n_time, n_pos)
    # Pre-allocate (time, pos, z) physical arrays and (time, pos) metadata arrays.
    values = {name: np.full(shape3, np.nan, dtype="f4") for name in PHYSICAL}
    lon = np.full(shape2, np.nan, dtype="f4")
    lat = np.full(shape2, np.nan, dtype="f4")
    pmax = np.full(shape2, np.nan, dtype="f4")
    time_1d = np.full(shape2, np.datetime64("NaT"), dtype="datetime64[ns]")
    mask = np.zeros(shape2, dtype=np.int8)
    mode = np.full(shape2, None, dtype=object)
    source_id = np.full(shape2, None, dtype=object)
    template = np.full(shape2, None, dtype=object)
    wmo = np.full(shape2, -1, dtype="i4")
    cyc = np.full(shape2, -1, dtype="i4")
    cycle = np.full(shape2, -1, dtype="i2")
    month_frac = np.full(shape2, np.nan, dtype="f4")
    edges = np.arange(0.0, 2020.0, 20.0)                                         # 20 m depth bin edges
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_index = np.digitize(depth, edges, right=True) - 1                        # depth level -> bin index
    nearest_bin = np.abs(depth[:, None] - centers[None, :]).argmin(axis=1)       # bin -> nearest native depth level

    for ti, (_, profiles) in enumerate(rows.groupby("target_month", sort=True)):
        profiles = profiles.sort_values("virtual_profile_rank_in_month")
        positions = profiles.virtual_profile_rank_in_month.to_numpy(dtype=int)
        if not np.array_equal(positions, np.arange(len(positions))):
            raise ValueError("Non-consecutive virtual-profile ranks")
        n = len(profiles)
        t_field = temperature.votemper.isel(time_counter=ti, deptht=slice(0, n_z)).values   # (z, y, x)
        s_field = salinity.vosaline.isel(time_counter=ti, deptht=slice(0, n_z)).values
        sampled_t = np.empty((n, n_z), dtype="f4")
        sampled_s = np.empty((n, n_z), dtype="f4")
        # Sample T and S at each virtual-profile position: bilinear stencil or nearest-eligible fallback.
        for i, row in enumerate(profiles.itertuples()):
            if row.sampling_mode == "bilinear":
                y0, y1 = int(row.bilin_iy0), int(row.bilin_iy1)
                x0, x1 = int(row.bilin_ix0), int(row.bilin_ix1)
                weights = (float(row.bilin_w00), float(row.bilin_w10),
                           float(row.bilin_w01), float(row.bilin_w11))
                for field, out in ((t_field, sampled_t), (s_field, sampled_s)):
                    out[i] = (weights[0] * field[:, y0, x0] + weights[1] * field[:, y0, x1]
                              + weights[2] * field[:, y1, x0] + weights[3] * field[:, y1, x1])
            else:
                y, x = int(row.ref_t_y), int(row.ref_t_x)
                sampled_t[i] = t_field[:, y, x]
                sampled_s[i] = s_field[:, y, x]

        sample_lat = profiles.sample_lat.to_numpy(dtype=float)
        sample_lon = profiles.sample_lon.to_numpy(dtype=float)
        assigned = profiles.assigned_max_pres_good.to_numpy(dtype=float)
        pressure = np.vstack([gsw.p_from_z(-depth, latitude) for latitude in sample_lat]).astype("f4")
        active = pressure <= assigned[:, None]                                   # levels within the assigned pressure limit
        sampled_t = np.where(active, sampled_t, np.nan).astype("f4")
        sampled_s = np.where(active, sampled_s, np.nan).astype("f4")
        lon2d = np.repeat(sample_lon[:, None], n_z, axis=1)
        lat2d = np.repeat(sample_lat[:, None], n_z, axis=1)
        # TEOS-10 chain: SP -> SA, (SA, in-situ T, p) -> CT, (SA, CT, p) -> rho.
        sa = gsw.SA_from_SP(sampled_s.astype(float), pressure.astype(float), lon2d, lat2d).astype("f4")
        ct = gsw.CT_from_t(sa.astype(float), sampled_t.astype(float), pressure.astype(float)).astype("f4")
        rho = gsw.rho(sa.astype(float), ct.astype(float), pressure.astype(float)).astype("f4")
        # Average rho in 20 m depth bins per profile, then map each bin back to the nearest native level.
        binned = np.full((n, len(centers)), np.nan, dtype="f4")
        count = np.zeros((n, len(centers)), dtype="i2")
        for i in range(n):
            for level, bin_number in enumerate(bin_index):
                value = rho[i, level]
                if np.isfinite(value) and 0 <= bin_number < len(centers):
                    if np.isnan(binned[i, bin_number]):
                        binned[i, bin_number] = value
                    else:
                        binned[i, bin_number] += value
                    count[i, bin_number] += 1
        nonempty = count > 0
        binned[nonempty] /= count[nonempty]
        for name, chunk in (("temperature", sampled_t), ("salinity", sampled_s),
                            ("SA", sa), ("CT", ct), ("rho", binned[:, nearest_bin])):
            values[name][ti, :n] = chunk
        lon[ti, :n], lat[ti, :n] = sample_lon.astype("f4"), sample_lat.astype("f4")
        pmax[ti, :n] = assigned.astype("f4")
        time_1d[ti, :n] = temperature.time_counter.values[ti]
        mask[ti, :n] = 1
        mode[ti, :n] = profiles.sampling_mode.to_numpy()
        source_id[ti, :n] = profiles.source_profile_id.to_numpy()
        template[ti, :n] = profiles.template_month.to_numpy()
        wmo[ti, :n], cyc[ti, :n] = profiles.wmo.to_numpy(dtype="i4"), profiles.cyc.to_numpy(dtype="i4")
        cycle[ti, :n] = profiles.cycle_id.to_numpy(dtype="i2")
        month_frac[ti, :n] = np.float32(0.5)
    pressure_axis = gsw.p_from_z(-depth, float(np.median(rows.sample_lat.to_numpy()))).astype("f4")
    ds = xr.Dataset(
        coords={"time": ("time", temperature.time_counter.values),
                "pos": ("pos", np.arange(n_pos, dtype="i4")),
                "z": ("z", depth.astype("f4")), "pressure": ("z", pressure_axis)},
        data_vars={**{name: (("time", "pos", "z"), array) for name, array in values.items()},
                   "lon": (("time", "pos"), lon), "lat": (("time", "pos"), lat),
                   "time_1d": (("time", "pos"), time_1d),
                   "profile_mask": (("time", "pos"), mask),
                   "assigned_max_pres_good": (("time", "pos"), pmax),
                   "sampling_mode": (("time", "pos"), mode),
                   "source_profile_id": (("time", "pos"), source_id),
                   "wmo": (("time", "pos"), wmo), "cyc": (("time", "pos"), cyc),
                   "cycle_id": (("time", "pos"), cycle),
                   "template_month": (("time", "pos"), template),
                   "month_frac": (("time", "pos"), month_frac)},
        attrs={"description": "Virtual Argo density input sampled from ORAS5 34.5S A-Z band",
               "sampling_geometry": "bilinear on eligible T cells; nearest eligible fallback",
               "vertical_masking": "QC max_pres_good shuffled within template month and applied in pressure space",
               "vertical_smoothing": "20 m mean bins on density, nearest-bin back-projection"},
    )
    if int(ds.profile_mask.values.sum()) != len(rows):
        raise ValueError("Monthly profile count mismatch")
    return annotate(ds)


def quarterly_input(monthly: xr.Dataset) -> xr.Dataset:
    """Concatenate every three consecutive monthly rows into one calendar-quarter sample."""
    mask = monthly.profile_mask.values.astype(bool)
    n_sample = monthly.sizes["time"] // 3                                        # one sample per non-overlapping 3-month block
    counts = mask.reshape(n_sample, 3, -1).sum(axis=(1, 2))
    n_pos = int(counts.max())
    n_z = monthly.sizes["z"]
    shape2 = (n_sample, n_pos)
    result = {}
    # Allocate one array per output variable with the matching dtype.
    for name, da in monthly.data_vars.items():
        if da.dims == ("time", "pos", "z"):
            arr = np.full((n_sample, n_pos, n_z), np.nan, dtype=da.dtype)
        elif name == "time_1d":
            arr = np.full(shape2, np.datetime64("NaT"), dtype="datetime64[ns]")
        elif da.dtype.kind in "OUS":
            arr = np.full(shape2, None, dtype=object)
        elif name == "profile_mask":
            arr = np.zeros(shape2, dtype=np.int8)
        elif name in ("wmo", "cyc", "cycle_id"):
            arr = np.full(shape2, -1, dtype=da.dtype)
        else:
            arr = np.full(shape2, np.nan, dtype=da.dtype)
        result[name] = arr
    month_in_quarter = np.full(shape2, -99, dtype=np.int8)
    # Fill each quarter by appending the valid positions from its three months.
    for sample in range(n_sample):
        end = 0
        for month in range(3):
            source = sample * 3 + month
            positions = np.flatnonzero(mask[source])
            stop = end + len(positions)
            for name in result:
                result[name][sample, end:stop] = monthly[name].values[source, positions]
            month_in_quarter[sample, end:stop] = month
            end = stop
    result["month_in_quarter"] = month_in_quarter
    quarter_time = monthly.time.values.reshape(-1, 3)[:, 1]                      # timestamp at middle month
    labels = np.array([f"{t.astype('datetime64[Y]').astype(int) + 1970}-Q{int(t.astype('datetime64[M]').astype(int) % 12 // 3 + 1)}"
                       for t in quarter_time], dtype=object)
    return annotate(xr.Dataset(
        coords={"sample": ("sample", quarter_time),
                "pos": ("pos", np.arange(n_pos, dtype="i4")),
                "z": ("z", monthly.z.values), "pressure": ("z", monthly.pressure.values)},
        data_vars={**{name: (("sample", "pos", "z") if arr.ndim == 3 else ("sample", "pos"), arr)
                      for name, arr in result.items()},
                   "quarter_label": (("sample",), labels)},
        attrs={"description": "Calendar-quarter virtual Argo profile cloud from three consecutive monthly inputs",
               "sample_time_definition": "Middle-month ORAS5 timestamp"},
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--qc-csv", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--salinity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    monthly_path, quarterly_path = args.output_dir / MONTHLY, args.output_dir / QUARTERLY
    if monthly_path.exists() or quarterly_path.exists():
        raise FileExistsError("Monthly or quarterly Argo output exists")
    rows = assigned_pressure(pd.read_parquet(args.geometry), pd.read_csv(args.qc_csv))
    with xr.open_dataset(args.temperature) as temperature, xr.open_dataset(args.salinity) as salinity:
        monthly = monthly_input(rows, temperature, salinity)
    quarterly = quarterly_input(monthly)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ds, path in ((monthly, monthly_path), (quarterly, quarterly_path)):
        temporary = path.with_name(path.name + ".partial")                       # staging file
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            ds.to_netcdf(temporary, engine="netcdf4", encoding={
                name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}
                for name in PHYSICAL
            })
            temporary.replace(path)                                              # rename when fully written
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Wrote {path}: {dict(ds.sizes)}", flush=True)


if __name__ == "__main__":
    main()
