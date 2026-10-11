#!/usr/bin/env python3
"""Build the 1958-2025 regional ORAS5 3-D salinity section at ~34.5 S.

Historical months come from ICDC OPeNDAP; recent global months come from
CDS or an existing raw directory. Each month is written directly to NetCDF
so the 75-level series does not need to fit in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import netCDF4
import numpy as np
import xarray as xr


VARIABLE = "vosaline"                     # ORAS5 variable name for salinity
CDS_VARIABLE = "salinity"                 # CDS variable key for the 2019-2025 fetch
PREFIX = "oras5_vosaline_34p5S_pm3deg"
DESCRIPTION = "ORAS5 3D salinity band around ~34.5S +/- 3.0 deg"
Y_START, Y_STOP = 336, 366                # ORCA025 rows for the 3-degree band (exclusive stop)
X_START, X_STOP = 933, 1228               # ORCA025 columns for the Atlantic longitudes
GRID_ATOL_DEG = 1e-5                      # tolerance for grid-equality checks
ICDC_ROOT = "https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit"
PERIODS = {
    "1958-1978": (1958, 1978, "ICDC backward extension"),
    "1979-2018": (1979, 2018, "ICDC standard ORAS5"),
    "2019-2025": (2019, 2025, "CDS ORAS5"),
}


def recent_filename(year: int, month: int) -> str:
    return f"{VARIABLE}_control_monthly_highres_3D_{year:04d}{month:02d}_OPER_v0.1.nc"


def download_recent_month(year: int, month: int, raw_dir: Path, client: object) -> None:
    """Download one 2019-2025 global ORAS5 salinity file from CDS to raw_dir."""
    destination = raw_dir / recent_filename(year, month)
    if destination.is_file():
        return
    if destination.exists():
        raise FileExistsError(destination)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{destination.name}.zip.partial"    # downloaded CDS zip
    temporary = raw_dir / f"{destination.name}.partial"      # extracted NetCDF
    archive.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    request = {
        "product_type": ["operational"],
        "vertical_resolution": ["all_levels"],
        "variable": [CDS_VARIABLE],
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
                        f"{VARIABLE}_ORAS5_1m_{year:04d}{month:02d}_grid_T_02.nc"
                    )
                sources.append((year, month, source))
    missing = [source for _, _, source in sources if isinstance(source, Path) and not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} monthly {VARIABLE} files; first: {missing[0]}")
    return sources


def create_output(path: Path, stage: str, n_months: int, part: xr.Dataset) -> netCDF4.Dataset:
    """Create the empty NetCDF for the full section and preallocate every variable."""
    output = netCDF4.Dataset(path, "w", format="NETCDF4")
    output.createDimension("time_counter", n_months)
    output.createDimension("deptht", 75)
    output.createDimension("y", 30)
    output.createDimension("x", 295)
    field = output.createVariable(VARIABLE, "f4", ("time_counter", "deptht", "y", "x"),
                                  fill_value=np.float32(9.96921e36), chunksizes=(1, 75, 30, 295))
    field.setncatts({**part[VARIABLE].attrs, "missing_value": np.float32(9.96921e36),
                     "coordinates": "nav_lat nav_lon"})
    for name, dimensions in (("nav_lat", ("y", "x")), ("nav_lon", ("y", "x")),
                             ("deptht", ("deptht",))):
        variable = output.createVariable(name, "f4", dimensions, fill_value=np.float32(np.nan))
        variable.setncatts(part[name].attrs)
        variable[:] = part[name].values
    clock = output.createVariable("time_counter", "i8", ("time_counter",))
    clock.units = "hours since 1958-01-16"
    clock.calendar = "proleptic_gregorian"
    output.setncatts({
        "description": DESCRIPTION,
        "y_start_global": Y_START,
        "y_end_global": Y_STOP - 1,
        "x_start_global": X_START,
        "x_end_global": X_STOP - 1,
        "lat_target_deg": -34.5,
        "lat_half_width_deg": 3.0,
        "lat_min_deg": -37.5,
        "lat_max_deg": -31.5,
        "n_time": n_months,
        "n_deptht": 75,
        "n_y": 30,
        "n_x": 295,
    })
    if stage == "all":
        output.source_segments = (
            "ICDC backward extension (1958-1978), ICDC standard ORAS5 (1979-2018), "
            "CDS ORAS5 (2019-2025)"
        )
    else:
        output.source_segment = PERIODS[stage][2]
    return output


def build(stage: str, raw_dir: Path, output_dir: Path) -> Path:
    """Stream every monthly file into the preallocated NetCDF and return its path."""
    sources = monthly_sources(stage, raw_dir)
    period_name = "1958_2025" if stage == "all" else stage.replace("-", "_")
    destination = output_dir / f"{PREFIX}_{period_name}.nc"
    if destination.exists():
        raise FileExistsError(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")   # staging file
    if temporary.exists():
        raise FileExistsError(temporary)
    output = None
    reference_lat = reference_lon = reference_depth = None
    first_time = last_time = None
    try:
        for index, (year, month, source) in enumerate(sources):
            # Open each month with up to three retries on transient OPeNDAP errors.
            for attempt in range(3):
                try:
                    with xr.open_dataset(source, engine="netcdf4") as dataset:
                        if dataset.sizes.get("time_counter") != 1:
                            raise ValueError(f"Expected one month in {source}")
                        part = dataset[[VARIABLE, "nav_lat", "nav_lon", "deptht"]].isel(
                            time_counter=0, y=slice(Y_START, Y_STOP), x=slice(X_START, X_STOP)
                        ).load()                                                           # (deptht, y, x) section
                        if index == 0:
                            rows = np.flatnonzero((dataset.nav_lat.mean("x").values >= -37.5)
                                                  & (dataset.nav_lat.mean("x").values <= -31.5))
                            if not np.array_equal(rows, np.arange(Y_START, Y_STOP)):
                                raise ValueError(f"Unexpected latitude rows: {rows.tolist()}")
                    break
                except OSError:
                    if attempt == 2 or isinstance(source, Path):
                        raise
                    time.sleep(attempt + 1)
            if dict(part.sizes) != {"deptht": 75, "y": 30, "x": 295}:
                raise ValueError(f"Unexpected section shape in {source}: {dict(part.sizes)}")
            date = np.asarray(part.time_counter.values).astype("datetime64[ns]")
            if date.astype("datetime64[M]") != np.datetime64(f"{year:04d}-{month:02d}", "M"):
                raise ValueError(f"Unexpected month in {source}: {date}")
            if output is None:
                # First month: open the output file and store the reference grid.
                output = create_output(temporary, stage, len(sources), part)
                reference_lat = part.nav_lat.values.copy()
                reference_lon = part.nav_lon.values.copy()
                reference_depth = part.deptht.values.copy()
                first_time = str(date)
            elif not (np.allclose(part.nav_lat.values, reference_lat, rtol=0, atol=GRID_ATOL_DEG)
                      and np.allclose(part.nav_lon.values, reference_lon, rtol=0, atol=GRID_ATOL_DEG)
                      and np.array_equal(part.deptht.values, reference_depth)):
                raise ValueError(f"Grid or depth changed in {source}")
            values = part[VARIABLE].values
            output[VARIABLE][index] = np.where(np.isfinite(values), values, np.float32(9.96921e36))
            output["time_counter"][index] = int(                                           # hours since 1958-01-16
                (date.astype("datetime64[h]") - np.datetime64("1958-01-16T00", "h"))
                / np.timedelta64(1, "h")
            )
            last_time = str(date)
            if month == 12:
                print(f"{stage}: {year} complete", flush=True)
        output.time_coverage_start = first_time
        output.time_coverage_end = last_time
        output.close()
        output = None
        temporary.replace(destination)                                                     # rename when fully written
    finally:
        if output is not None:
            output.close()
        temporary.unlink(missing_ok=True)
    print(f"Wrote {destination}: {len(sources)} months", flush=True)
    return destination


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
    build(args.stage, args.recent_raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
