#!/usr/bin/env python3
"""Download frozen GDAC Argo candidates and measure per-profile QC metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


BASE_URL = "https://data-argo.ifremer.fr/dac"
COLUMNS = ("wmo", "cyc", "date", "lat", "lon", "file", "url", "http_status", "error",
           "POSITION_QC", "JULD_QC", "DATA_MODE", "PROFILE_PRES_QC", "PROFILE_TEMP_QC",
           "PROFILE_PSAL_QC", "PRES_QC_SOURCE", "TEMP_QC_SOURCE", "PSAL_QC_SOURCE",
           "n_levels_0_2000", "n_good_pres_0_2000", "n_good_temp_0_2000",
           "n_good_psal_0_2000", "frac_good_temp_0_2000", "frac_good_psal_0_2000",
           "max_pres_good")


def decode(values: np.ndarray) -> np.ndarray:
    """Convert a numpy array of bytes-or-string items into a plain object array of strings."""
    return np.array([item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
                     for item in np.asarray(values).ravel()], dtype=object)


def inspect_profile(file: Path) -> dict:
    """Open one Argo profile file and return QC counts and fractions over the 0-2000 dbar range."""
    with xr.open_dataset(file) as ds:
        row = {
            "POSITION_QC": decode(ds.POSITION_QC.values[0])[0],
            "JULD_QC": decode(ds.JULD_QC.values[0])[0],
            "DATA_MODE": decode(ds.DATA_MODE.values[0])[0] if "DATA_MODE" in ds else None,
        }
        for field in ("PRES", "TEMP", "PSAL"):
            name = f"PROFILE_{field}_QC"
            row[name] = decode(ds[name].values[0])[0] if name in ds else None
        pressure_name = "PRES_ADJUSTED" if "PRES_ADJUSTED" in ds else "PRES"
        pressure = np.asarray(ds[pressure_name].values[0, :])
        codes = {}
        for field in ("PRES", "TEMP", "PSAL"):
            preferred = f"{field}_ADJUSTED_QC"            # prefer delayed-mode adjusted QC when available
            name = preferred if preferred in ds else f"{field}_QC"
            row[f"{field}_QC_SOURCE"] = name
            codes[field] = decode(ds[name].values[0, :])
        in_range = np.isfinite(pressure) & (pressure >= 0) & (pressure <= 2000)   # levels inside 0-2000 dbar
        good_pres = np.isin(codes["PRES"], ["1", "2"]) & in_range                 # Argo QC codes 1 (good) and 2 (probably good)
        good_temp = np.isin(codes["TEMP"], ["1", "2"]) & good_pres
        good_psal = np.isin(codes["PSAL"], ["1", "2"]) & good_pres
        n_levels = int(in_range.sum())
        row.update({
            "n_levels_0_2000": n_levels,
            "n_good_pres_0_2000": int(good_pres.sum()),
            "n_good_temp_0_2000": int(good_temp.sum()),
            "n_good_psal_0_2000": int(good_psal.sum()),
            "frac_good_temp_0_2000": int(good_temp.sum()) / n_levels if n_levels else np.nan,
            "frac_good_psal_0_2000": int(good_psal.sum()) / n_levels if n_levels else np.nan,
            "max_pres_good": float(np.nanmax(pressure[good_pres])) if good_pres.any() else np.nan,
        })
    return row


def inspect_candidate(candidate: object, cache_dir: Path, session: requests.Session) -> dict:
    """Download the candidate profile if missing and return its QC row."""
    relative = str(candidate.file)
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe GDAC path: {relative}")
    url = f"{BASE_URL}/{relative}"
    cached = cache_dir / Path(relative).name
    row = {"wmo": int(candidate.wmo), "cyc": int(candidate.cyc),
           "date": candidate.date, "lat": float(candidate.lat),
           "lon": float(candidate.lon), "file": relative, "url": url,
           "http_status": None, "error": None}
    try:
        if not cached.exists():
            with session.get(url, stream=True, timeout=60) as response:
                row["http_status"] = response.status_code
                response.raise_for_status()
                partial = cached.with_name(cached.name + ".partial")   # staging file
                try:
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                    partial.replace(cached)                            # rename when fully written
                finally:
                    partial.unlink(missing_ok=True)
        else:
            row["http_status"] = 200
        row.update(inspect_profile(cached))
    except Exception as error:
        row["error"] = str(error)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry existing non-404 failures without revisiting successful rows")
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    required = ["wmo", "cyc", "date", "lat", "lon", "file"]
    if candidates[required].isna().any().any() or candidates.duplicated(["wmo", "cyc"]).any():
        raise ValueError("Candidate manifest has missing or duplicate profiles")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    # Resume support: an existing output must be a prefix of the candidate manifest.
    completed = 0
    previous = None
    if args.output.exists():
        previous = pd.read_csv(args.output, low_memory=False)
        completed = len(previous)
        if not previous[["wmo", "cyc"]].reset_index(drop=True).equals(
            candidates.loc[:completed - 1, ["wmo", "cyc"]].reset_index(drop=True)
        ):
            raise ValueError("Existing output is not a prefix of the candidate manifest")
    if args.retry_errors and (previous is None or completed != len(candidates)):
        parser.error("--retry-errors requires a complete existing catalog")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        if args.retry_errors:
            # Retry only rows that recorded a non-404 error; 404 rows are considered final.
            retry = previous.index[previous.error.notna() & previous.http_status.ne(404)]
            print(f"Retrying {len(retry)} non-404 failures", flush=True)
            for column in ("POSITION_QC", "JULD_QC", "DATA_MODE", "PROFILE_PRES_QC",
                           "PROFILE_TEMP_QC", "PROFILE_PSAL_QC"):
                previous[column] = previous[column].astype(object)
            for count, index in enumerate(retry, 1):
                row = inspect_candidate(candidates.iloc[index], args.cache_dir, session)
                for column in COLUMNS:
                    previous.at[index, column] = row.get(column)
                if count % args.batch_size == 0:
                    print(f"Retried {count}/{len(retry)}", flush=True)
            temporary = args.output.with_name(args.output.name + ".retry.partial")
            if temporary.exists():
                raise FileExistsError(temporary)
            try:
                previous.to_csv(temporary, index=False)
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            # Append new rows in batches so a crash only loses the current batch.
            rows = []
            for candidate in candidates.iloc[completed:].itertuples(index=False):
                rows.append(inspect_candidate(candidate, args.cache_dir, session))
                if len(rows) >= args.batch_size:
                    pd.DataFrame(rows, columns=COLUMNS).to_csv(
                        args.output, mode="a", header=not args.output.exists(), index=False)
                    completed += len(rows)
                    print(f"Processed {completed}/{len(candidates)}", flush=True)
                    rows.clear()
            if rows:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(
                    args.output, mode="a", header=not args.output.exists(), index=False)
                completed += len(rows)
    print(f"Wrote {args.output}: {completed} profiles")


if __name__ == "__main__":
    main()
