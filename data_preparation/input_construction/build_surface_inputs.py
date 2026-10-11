#!/usr/bin/env python3
"""Build the four monthly surface inputs at the four A-Z scales.

SST, SSS and SSH use T-cell area weights. OBP uses a GRACE-like 300-km-FWHM
Gaussian filter and unweighted ocean-cell means. Quarterly models consume
three consecutive rows of these monthly files; no quarterly product is made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr


ROOT = Path(__file__).resolve().parents[1] / "data"
SCALES = ((20, 20), (40, 10), (80, 5), (160, 3))      # (n_bins, n_lat_rows)
A_LON, Z_LON = -51.5, 17.557667                       # T-grid bin domain, not the V-grid target endpoint
FIELDS = {
    "sst": ("SST", "oras5_sosstsst_34p5S_pm2deg_1958_2025.nc", "sosstsst", "sst_bin_means", "oras5_sst_input_AZ"),
    "sss": ("SSS", "oras5_sosaline_34p5S_pm2deg_1958_2025.nc", "sosaline", "sss_bin_means", "oras5_sss_input_AZ"),
    "ssh": ("SSH", "oras5_sossheig_34p5S_pm2deg_1958_2025.nc", "sossheig", "ssh_bin_means", "oras5_ssh_input_AZ"),
    "obp": ("OBP", "oras5_obp_gracelike_34p5S_pm2deg_1958_2025_ref2004_2009_fixedints.nc", "obp_anom_eqw_ref_2004_2009", "obp_bin_means", "oras5_virtual_wlt_input_AZ"),
}


def filtered_obp(source: xr.Dataset) -> np.ndarray:
    """Apply the 300 km FWHM Gaussian filter to the OBP anomaly on the ocean mask."""
    base = source["obp_anom_eqw_ref_2004_2009"].values.astype(np.float64) * 100.0   # equivalent water thickness in cm
    ocean = source["tmask_surface"].values == 1
    out = np.empty(base.shape, dtype=np.float32)
    for t, image in enumerate(base):
        valid = np.isfinite(image)
        # Land-aware normalisation: numerator and denominator share the same Gaussian kernel.
        numerator = gaussian_filter(np.where(valid, image, 0.0), (5.565799, 5.565859), mode="nearest", truncate=4.0)
        denominator = gaussian_filter(valid.astype(np.float64), (5.565799, 5.565859), mode="nearest", truncate=4.0)
        out[t] = np.where(ocean & (denominator > 1e-12), numerator / np.where(denominator > 1e-12, denominator, 1), np.nan)
    return out


def geometry(source: xr.Dataset, mesh: xr.Dataset | None, n_bins: int, n_rows: int):
    """Pick the n_rows central latitudes, assign each column to one of n_bins, and return per-bin weights."""
    lat = source["nav_lat"].values[:, 0].astype(np.float64)
    rows = np.sort(np.argsort(np.abs(lat + 34.5))[:n_rows])        # the n_rows latitudes closest to 34.5 S
    lon = source["nav_lon"].values
    if mesh is None:
        # OBP branch: no mesh, use the dataset's own ocean mask and unweighted ocean-cell means.
        ocean = source["tmask_surface"].values == 1
        selected = np.zeros(lat.size, dtype=bool)
        selected[rows] = True
        az = ocean & selected[:, None] & (lon >= A_LON) & (lon <= Z_LON)
        column_lon = np.array([np.mean(lon[az[:, x], x]) if az[:, x].any() else np.nan for x in range(lon.shape[1])])
        edges = np.linspace(A_LON, Z_LON, n_bins + 1)
        bin_id = np.digitize(column_lon, edges[1:-1])
        count = np.array([(az & (bin_id[None, :] == b)).sum() for b in range(n_bins)], dtype=np.int32)
        return rows, az, bin_id, edges, count, None

    # SST/SSS/SSH branch: use mesh T-cell areas as weights.
    x = np.where((lon[0] >= A_LON) & (lon[0] <= Z_LON))[0]
    edges = np.linspace(float(lon[0, x[0]]), float(lon[0, x[-1]]), n_bins + 1)
    bin_id = np.digitize(lon[0, x], edges[1:-1])
    y0, x0 = int(source.attrs["y_start_global"]), int(source.attrs["x_start_global"])
    yg, xg = y0 + rows, x0 + x
    ocean = mesh["tmaskutil"].isel(t=0).values[np.ix_(yg, xg)] == 1
    area = mesh["e1t"].isel(t=0).values[np.ix_(yg, xg)] * mesh["e2t"].isel(t=0).values[np.ix_(yg, xg)]
    weight = np.where(ocean, area, 0.0)                              # zero weight on land cells
    count = np.array([ocean[:, bin_id == b].sum() for b in range(n_bins)], dtype=np.int32)
    return rows, x, bin_id, edges, count, weight


def build_field(field: str, scales: list[tuple[int, int]], overwrite: bool, root: Path = ROOT) -> None:
    """Build every requested scale for one surface field and write one NetCDF per scale."""
    subdir, name, source_var, output_var, prefix = FIELDS[field]
    path = root / "dynamic" / subdir / name
    with xr.open_dataset(path) as source:
        time = source["time_counter"].values
        if len(time) != 816:
            raise ValueError(f"{field}: expected 816 months, found {len(time)}")
        mesh_path = root / "static" / "mesh_mask.nc"
        mesh = xr.open_dataset(mesh_path, decode_times=False) if field != "obp" else None
        try:
            values = filtered_obp(source) if field == "obp" else source[source_var].values.astype(np.float64)
            for n_bins, n_rows in scales:
                rows, columns, bin_id, edges, count, weight = geometry(source, mesh, n_bins, n_rows)
                if (count <= 0).any():
                    raise ValueError(f"{field} {n_bins}: empty ocean bin")
                means = np.empty((816, n_bins), dtype=np.float64)
                area_sum = None
                if field == "obp":
                    for b in range(n_bins):
                        mask = columns & (bin_id[None, :] == b)
                        if n_bins == 20 and n_rows == 20:
                            # 20-bin case: per-month reduction over finite values.
                            for t in range(len(time)):
                                finite = values[t][mask]
                                finite = finite[np.isfinite(finite)]
                                means[t, b] = np.mean(finite)
                        else:
                            means[:, b] = np.nanmean(values[:, mask], axis=1)
                else:
                    selected = values[:, rows, :][:, :, columns]
                    area_sum = np.array([weight[:, bin_id == b].sum() for b in range(n_bins)])
                    # Area-weighted mean over the ocean cells inside each bin.
                    for b in range(n_bins):
                        v, w = selected[:, :, bin_id == b], weight[:, bin_id == b]
                        valid = np.isfinite(v) & (w[None] > 0)
                        means[:, b] = np.sum(np.where(valid, v * w[None], 0), axis=(1, 2)) / np.sum(np.where(valid, w[None], 0), axis=(1, 2))
                if not np.isfinite(means).all():
                    raise ValueError(f"{field} {n_bins}: non-finite bin mean")
                data = {output_var: (("time_counter", "bin"), means.astype(np.float32)),
                        "bin_edge_left": ("bin", edges[:-1].astype(np.float32)),
                        "bin_edge_right": ("bin", edges[1:].astype(np.float32)),
                        "bin_lon_center": ("bin", (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)),
                        "bin_n_ocean_cells": ("bin", count)}
                if area_sum is not None:
                    data["bin_area_sum"] = ("bin", area_sum.astype(np.float64))
                ds = xr.Dataset(data, coords={"time_counter": time, "bin": np.arange(n_bins, dtype=np.int32)},
                                attrs={"surface_lat_rows": n_rows, "selected_lat_row_indices_local": ",".join(map(str, rows)),
                                       "A_longitude_deg": A_LON, "Z_longitude_deg": Z_LON,
                                       "feature_kind": "bin_means", "n_bins": n_bins, "n_features": n_bins,
                                       "source_variable": source_var})
                ds[output_var].attrs["units"] = "cm" if field == "obp" else {"sst": "degC", "sss": "psu", "ssh": "m"}[field]
                out = root / "input" / subdir / f"{prefix}_{n_bins}bin{'_lat' + str(n_rows) if n_rows != 20 else ''}_means_1958_2025.nc"
                if out.exists() and not overwrite:
                    raise FileExistsError(out)
                out.parent.mkdir(parents=True, exist_ok=True)
                temporary = out.with_suffix(".nc.partial")   # staging file
                try:
                    ds.to_netcdf(temporary)
                    temporary.replace(out)                   # rename when fully written
                finally:
                    temporary.unlink(missing_ok=True)
                print(out, flush=True)
        finally:
            if mesh is not None:
                mesh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", choices=FIELDS, default=list(FIELDS))
    parser.add_argument("--scales", nargs="+", type=int, choices=[20, 40, 80, 160], default=[20, 40, 80, 160])
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Data tree for inputs and outputs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scales = [scale for scale in SCALES if scale[0] in args.scales]
    for field in args.fields:
        build_field(field, scales, args.overwrite, args.data_root)


if __name__ == "__main__":
    main()
