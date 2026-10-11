#!/usr/bin/env python3
"""Build the 1958-2025 ORAS5 SSH section at ~34.5 S, one period at a time.

Historical monthly fields are read from ICDC OPeNDAP; recent monthly fields
come from the original global ORAS5 files, optionally retrieved from CDS.
Run --stage all in an empty output directory to keep only the final NetCDF,
or run stages separately to inspect the three intermediate periods.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import numpy as np
import xarray as xr


VARIABLE = "sossheig"                     # ORAS5 variable name for SSH
PREFIX = "oras5_sossheig_34p5S_pm2deg"
LAT_TARGET = -34.5                        # band center in degrees
LAT_HALF_WIDTH = 2.0                      # band half-width in degrees
Y_START, Y_STOP = 341, 361                # ORCA025 rows for the band (exclusive stop)
X_START, X_STOP = 933, 1228               # ORCA025 columns for the Atlantic longitudes
GRID_ATOL_DEG = 1e-5                      # tolerance for grid-equality checks
ICDC_ROOT = "https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit"
PERIODS = {
    "1958-1978": (1958, 1978, "ICDC backward extension"),
    "1979-2018": (1979, 2018, "ICDC standard ORAS5"),
    "2019-2025": (2019, 2025, "CDS ORAS5"),
}


def output_path(output_dir: Path, period: str) -> Path:
    return output_dir / f"{PREFIX}_{period.replace('-', '_')}.nc"


def recent_filename(year: int, month: int) -> str:
    return f"sossheig_control_monthly_highres_2D_{year:04d}{month:02d}_OPER_v0.1.nc"


def download_recent_month(year: int, month: int, raw_dir: Path, client: object) -> None:
    """Download one 2019-2025 global ORAS5 SSH file from CDS to raw_dir."""
    destination = raw_dir / recent_filename(year, month)
    if destination.is_file():
        return
    if destination.exists():
        raise FileExistsError(destination)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{destination.name}.zip.partial"    # downloaded CDS zip
    temporary = raw_dir / f"{destination.name}.partial"      # extracted NetCDF
    # Clean up partial files left by any earlier interrupted run.
    archive.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    request = {
        "product_type": ["operational"],
        "vertical_resolution": ["single_level"],
        "variable": ["sea_surface_height"],
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


def download_recent(raw_dir: Path) -> None:
    """Download all missing 2019-2025 monthly SSH files from CDS."""
    import cdsapi

    client = cdsapi.Client(quiet=True, progress=False, retry_max=5, sleep_max=15)
    for year in range(2019, 2026):
        for month in range(1, 13):
            download_recent_month(year, month, raw_dir, client)


def monthly_sources(period: str, recent_raw_dir: Path | None) -> list[tuple[int, int, str | Path]]:
    """List (year, month, source) for every month in the requested period."""
    first, last, _ = PERIODS[period]
    sources = []
    for year in range(first, last + 1):
        for month in range(1, 13):
            ym = f"{year:04d}{month:02d}"
            if period == "2019-2025":
                if recent_raw_dir is None:
                    raise ValueError("--recent-raw-dir is required for 2019-2025")
                source = recent_raw_dir / recent_filename(year, month)                   # local NetCDF path
            else:
                branch = "oras5_backward_extension" if period == "1958-1978" else "oras5"
                source = (                                                                # ICDC OPeNDAP URL
                    f"{ICDC_ROOT}/{branch}/ORCA025/sossheig/opa0/"
                    f"sossheig_ORAS5_1m_{ym}_grid_T_02.nc"
                )
            sources.append((year, month, source))
    if period == "2019-2025":
        missing = [path for _, _, path in sources if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} monthly SSH files; first: {missing[0]}")
    return sources


def load_month(source: str | Path, year: int, month: int) -> xr.Dataset:
    """Open one monthly file and return the ~34.5 S section as a single-time dataset."""
    for attempt in range(3):
        try:
            with xr.open_dataset(source, engine="netcdf4") as dataset:
                if dataset.sizes.get("time_counter") != 1:
                    raise ValueError(f"Expected one month in {source}")
                part = dataset[[VARIABLE, "nav_lat", "nav_lon"]].isel(
                    time_counter=0,
                    y=slice(Y_START, Y_STOP),
                    x=slice(X_START, X_STOP),
                ).load()                                                                 # (y, x) section
                if part.sizes != {"y": Y_STOP - Y_START, "x": X_STOP - X_START}:
                    raise ValueError(f"Unexpected SSH section shape in {source}: {dict(part.sizes)}")
                date = np.asarray(part.time_counter.values).astype("datetime64[M]")
                if date != np.datetime64(f"{year:04d}-{month:02d}", "M"):
                    raise ValueError(f"Unexpected SSH month in {source}: {date}")
                part = part.expand_dims("time_counter")                                  # keep time axis for concat
                part.time_counter.attrs.clear()
                return part
        except OSError:
            # Retry transient OPeNDAP errors up to three times.
            if attempt == 2 or isinstance(source, Path):
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def build_period(period: str, recent_raw_dir: Path | None = None) -> xr.Dataset:
    """Concatenate every monthly section in one period and verify the grid."""
    parts = []
    reference_lat = reference_lon = None
    for year, month, source in monthly_sources(period, recent_raw_dir):
        part = load_month(source, year, month)
        if reference_lat is None:
            # First month of the period: establish the reference grid.
            with xr.open_dataset(source, engine="netcdf4") as dataset:
                row_lat = dataset.nav_lat.mean(dim="x").values                           # one lat per row
            rows = np.flatnonzero(
                (row_lat >= LAT_TARGET - LAT_HALF_WIDTH)
                & (row_lat <= LAT_TARGET + LAT_HALF_WIDTH)
            )
            if not np.array_equal(rows, np.arange(Y_START, Y_STOP)):
                raise ValueError(f"Unexpected SSH latitude rows: {rows.tolist()}")
            reference_lat = part.nav_lat.values.copy()
            reference_lon = part.nav_lon.values.copy()
        elif not (np.allclose(part.nav_lat.values, reference_lat, rtol=0, atol=GRID_ATOL_DEG)
                  and np.allclose(part.nav_lon.values, reference_lon, rtol=0, atol=GRID_ATOL_DEG)):
            raise ValueError(f"SSH grid changed in {source}")
        parts.append(part)
        if month == 12:
            print(f"{period}: {year} complete", flush=True)

    result = xr.concat(parts, dim="time_counter", coords="minimal", compat="override")
    _, _, source_segment = PERIODS[period]
    result.attrs = section_attrs(result)
    result.attrs["source_segment"] = source_segment
    return result


def section_attrs(dataset: xr.Dataset) -> dict:
    """Return the NetCDF attributes common to every SSH section file."""
    return {
        "description": "ORAS5 sea surface height band around ~34.5S +/- 2.0 deg",
        "time_coverage_start": str(dataset.time_counter.values[0]),
        "time_coverage_end": str(dataset.time_counter.values[-1]),
        "y_start_global": Y_START,
        "y_end_global": Y_STOP - 1,
        "x_start_global": X_START,
        "x_end_global": X_STOP - 1,
        "lat_target_deg": LAT_TARGET,
        "lat_half_width_deg": LAT_HALF_WIDTH,
        "lat_min_deg": LAT_TARGET - LAT_HALF_WIDTH,
        "lat_max_deg": LAT_TARGET + LAT_HALF_WIDTH,
        "n_time": dataset.sizes["time_counter"],
        "n_y": dataset.sizes["y"],
        "n_x": dataset.sizes["x"],
    }


def join_periods(output_dir: Path) -> xr.Dataset:
    """Join the three period files into one continuous 1958-2025 section."""
    with ExitStack() as stack:
        segments = [
            stack.enter_context(xr.open_dataset(output_path(output_dir, period), engine="netcdf4"))
            for period in PERIODS
        ]
        # Check per-period month counts and that every segment shares the same grid.
        for period, segment in zip(PERIODS, segments):
            first, last, _ = PERIODS[period]
            if segment.sizes.get("time_counter") != 12 * (last - first + 1):
                raise ValueError(f"Wrong number of months in {period}")
            if not (np.allclose(segment.nav_lat.values, segments[0].nav_lat.values,
                                rtol=0, atol=GRID_ATOL_DEG)
                    and np.allclose(segment.nav_lon.values, segments[0].nav_lon.values,
                                    rtol=0, atol=GRID_ATOL_DEG)):
                raise ValueError(f"SSH grid differs in {period}")
        result = xr.concat(segments, dim="time_counter", coords="minimal", compat="override")
        expected = np.arange(np.datetime64("1958-01"), np.datetime64("2026-01"), dtype="datetime64[M]")
        if not np.array_equal(result.time_counter.values.astype("datetime64[M]"), expected):
            raise ValueError("SSH time series is not monthly and continuous from 1958 through 2025")
        result = result.load()
    result.attrs = section_attrs(result)
    result.attrs["source_segments"] = (
        "ICDC backward extension (1958-1978), ICDC standard ORAS5 (1979-2018), "
        "CDS ORAS5 (2019-2025)"
    )
    return result


def write_new(dataset: xr.Dataset, path: Path) -> None:
    """Write dataset to path atomically; refuse to overwrite an existing file."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")   # staging file
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        dataset.to_netcdf(temporary, engine="netcdf4")
        temporary.replace(path)                          # rename when fully written
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {path}: {dict(dataset.sizes)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*PERIODS, "join", "all"), default="all")
    parser.add_argument("--recent-raw-dir", type=Path, help="Directory of 84 global 2019-2025 SSH files")
    parser.add_argument("--download-recent", action="store_true", help="Fetch missing 2019-2025 months from CDS")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stages = (*PERIODS, "join") if args.stage == "all" else (args.stage,)
    if args.download_recent and (args.recent_raw_dir is None or "2019-2025" not in stages):
        parser.error("--download-recent requires --recent-raw-dir and --stage 2019-2025 or all")
    if "2019-2025" in stages and args.recent_raw_dir is None:
        parser.error("--recent-raw-dir is required for --stage 2019-2025 or all")
    if args.stage == "all":
        # --stage all starts from an empty output directory and keeps only the joined file.
        existing = [output_path(args.output_dir, period) for period in (*PERIODS, "1958-2025")]
        if any(path.exists() for path in existing):
            raise FileExistsError("--stage all requires an empty SSH output directory")
    if "2019-2025" in stages:
        if args.stage == "2019-2025" and output_path(args.output_dir, args.stage).exists():
            raise FileExistsError(output_path(args.output_dir, args.stage))
        if args.download_recent:
            download_recent(args.recent_raw_dir)
        else:
            monthly_sources("2019-2025", args.recent_raw_dir)
    for stage in stages:
        if stage == "join":
            result = join_periods(args.output_dir)
            path = output_path(args.output_dir, "1958-2025")
        else:
            path = output_path(args.output_dir, stage)
            if path.exists():
                raise FileExistsError(path)
            result = build_period(stage, args.recent_raw_dir)
        write_new(result, path)
    if args.stage == "all":
        # Only the joined 1958-2025 file is kept; the per-period files are removed.
        for period in PERIODS:
            output_path(args.output_dir, period).unlink()


if __name__ == "__main__":
    main()
