#!/usr/bin/env python3
"""Build the 1958-2025 ORAS5 zonal wind-stress band at ~34.5 S.

The output is a 15-row U-grid band across the Atlantic longitudes. Historical
months come from ICDC OPeNDAP; recent global months come from CDS or an
existing raw directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import numpy as np
import xarray as xr


VARIABLE = "sozotaux"                     # ORAS5 variable name for zonal wind stress
PREFIX = "oras5_sozotaux_34p5S_pm1p5deg"
Y_START, Y_STOP = 344, 359                # ORCA025 rows for the 1.5-degree band (exclusive stop)
X_START, X_STOP = 933, 1228               # ORCA025 columns for the Atlantic longitudes
ICDC_ROOT = "https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit"
PERIODS = {
    "1958-1978": (1958, 1978, "ICDC backward extension"),
    "1979-2018": (1979, 2018, "ICDC standard ORAS5"),
    "2019-2025": (2019, 2025, "CDS ORAS5"),
}


def recent_filename(year: int, month: int) -> str:
    return f"sozotaux_control_monthly_highres_2D_{year:04d}{month:02d}_OPER_v0.1.nc"


def download_recent_month(year: int, month: int, raw_dir: Path, client: object) -> None:
    """Download one 2019-2025 global ORAS5 zonal-wind-stress file from CDS to raw_dir."""
    destination = raw_dir / recent_filename(year, month)
    if destination.is_file():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{destination.name}.zip.partial"    # downloaded CDS zip
    temporary = raw_dir / f"{destination.name}.partial"      # extracted NetCDF
    archive.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    request = {
        "product_type": ["operational"],
        "vertical_resolution": ["single_level"],
        "variable": ["zonal_wind_stress"],
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
    }
    try:
        client.retrieve("reanalysis-oras5", request, str(archive))
        with ZipFile(archive) as zipped:
            if zipped.namelist() != [destination.name]:
                raise ValueError(f"Unexpected CDS archive contents for {year:04d}-{month:02d}")
            with zipped.open(destination.name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        temporary.replace(destination)                        # rename when fully written
    finally:
        archive.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {year:04d}-{month:02d}", flush=True)


def monthly_sources(stage: str, raw_dir: Path) -> list[tuple[int, int, str | Path]]:
    """List (year, month, source) for every month in the requested stage."""
    periods = PERIODS if stage == "all" else {stage: PERIODS[stage]}
    sources = []
    for period, (first, last, _) in periods.items():
        for year in range(first, last + 1):
            for month in range(1, 13):
                if period == "2019-2025":
                    source = raw_dir / recent_filename(year, month)                      # local NetCDF path
                else:
                    branch = "oras5_backward_extension" if period == "1958-1978" else "oras5"
                    source = (                                                           # ICDC OPeNDAP URL
                        f"{ICDC_ROOT}/{branch}/ORCA025/{VARIABLE}/opa0/"
                        f"{VARIABLE}_ORAS5_1m_{year:04d}{month:02d}_grid_U_02.nc"
                    )
                sources.append((year, month, source))
    missing = [path for _, _, path in sources if isinstance(path, Path) and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} recent wind-stress files; first: {missing[0]}")
    return sources


def build(stage: str, raw_dir: Path) -> xr.Dataset:
    """Concatenate every monthly U-band section into one dataset and verify the grid."""
    parts = []
    reference_lat = reference_lon = None
    for year, month, source in monthly_sources(stage, raw_dir):
        # Open each month with up to three retries on transient OPeNDAP errors.
        for attempt in range(3):
            try:
                with xr.open_dataset(source, engine="netcdf4") as dataset:
                    if dataset.sizes.get("time_counter") != 1:
                        raise ValueError(f"Expected one month in {source}")
                    part = dataset[[VARIABLE, "nav_lat", "nav_lon"]].isel(
                        time_counter=0, y=slice(Y_START, Y_STOP), x=slice(X_START, X_STOP)
                    ).load()                                                             # (y, x) section
                    if reference_lat is None:
                        rows = np.flatnonzero((dataset.nav_lat.mean("x").values >= -36.0)
                                              & (dataset.nav_lat.mean("x").values <= -33.0))
                        if not np.array_equal(rows, np.arange(Y_START, Y_STOP)):
                            raise ValueError(f"Unexpected U-grid latitude rows: {rows.tolist()}")
                break
            except OSError:
                if attempt == 2 or isinstance(source, Path):
                    raise
                time.sleep(attempt + 1)
        if dict(part.sizes) != {"y": 15, "x": 295}:
            raise ValueError(f"Unexpected wind-stress section shape in {source}")
        date = np.asarray(part.time_counter.values).astype("datetime64[M]")
        if date != np.datetime64(f"{year:04d}-{month:02d}", "M"):
            raise ValueError(f"Unexpected month in {source}: {date}")
        if reference_lat is None:
            reference_lat, reference_lon = part.nav_lat.values.copy(), part.nav_lon.values.copy()
        elif not (np.allclose(part.nav_lat.values, reference_lat, rtol=0, atol=1e-5)
                  and np.allclose(part.nav_lon.values, reference_lon, rtol=0, atol=1e-5)):
            raise ValueError(f"U-grid changed in {source}")
        part = part.expand_dims("time_counter")                                           # keep time axis for concat
        part.time_counter.attrs.clear()
        parts.append(part)
        if month == 12:
            print(f"{stage}: {year} complete", flush=True)

    result = xr.concat(parts, dim="time_counter", coords="minimal", compat="override")
    result.attrs = {
        "description": "ORAS5 zonal wind stress band around ~34.5S +/- 1.5 deg",
        "time_coverage_start": str(result.time_counter.values[0]),
        "time_coverage_end": str(result.time_counter.values[-1]),
        "y_start_global": Y_START, "y_end_global": Y_STOP - 1,
        "x_start_global": X_START, "x_end_global": X_STOP - 1,
        "lat_target_deg": -34.5, "lat_half_width_deg": 1.5,
        "lat_min_deg": -36.0, "lat_max_deg": -33.0,
        "n_time": result.sizes["time_counter"], "n_y": 15, "n_x": 295,
    }
    if stage == "all":
        result.attrs["source_segments"] = (
            "ICDC backward extension (1958-1978), ICDC standard ORAS5 (1979-2018), "
            "CDS ORAS5 (2019-2025)"
        )
    else:
        result.attrs["source_segment"] = PERIODS[stage][2]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*PERIODS, "all"), default="all")
    parser.add_argument("--recent-raw-dir", type=Path, required=True)
    parser.add_argument("--download-recent", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.download_recent and args.stage not in ("2019-2025", "all"):
        parser.error("--download-recent requires --stage 2019-2025 or all")
    if args.download_recent:
        import cdsapi

        client = cdsapi.Client(quiet=True, progress=False, retry_max=5, sleep_max=15)
        for year in range(2019, 2026):
            for month in range(1, 13):
                download_recent_month(year, month, args.recent_raw_dir, client)
    period = "1958_2025" if args.stage == "all" else args.stage.replace("-", "_")
    output = args.output_dir / f"{PREFIX}_{period}.nc"
    if output.exists():
        raise FileExistsError(output)
    result = build(args.stage, args.recent_raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")   # staging file
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        result.to_netcdf(temporary, engine="netcdf4")
        temporary.replace(output)                            # rename when fully written
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {output}: {dict(result.sizes)}", flush=True)


if __name__ == "__main__":
    main()
