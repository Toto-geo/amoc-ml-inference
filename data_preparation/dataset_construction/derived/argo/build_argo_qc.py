#!/usr/bin/env python3
"""Apply the strict Argo QC filter to a GDAC profile-metrics table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_token(value: object) -> str:
    """Strip the b'...' / b\"...\" wrapper that pandas can leave on Argo byte-string fields."""
    token = str(value).strip()
    if (token.startswith("b'") and token.endswith("'")) or (
        token.startswith('b"') and token.endswith('"')
    ):
        token = token[2:-1]
    return token.strip()


def build(all_profiles: pd.DataFrame) -> pd.DataFrame:
    """Normalise column types and apply the strict Argo QC filter."""
    qc = all_profiles.copy()
    # Normalise numeric and categorical columns before filtering.
    qc["wmo"] = pd.to_numeric(qc.wmo, errors="coerce").astype("Int64")
    qc["cyc"] = pd.to_numeric(qc.cyc, errors="coerce").astype("Int64")
    qc["date"] = pd.to_datetime(qc.date, errors="coerce")
    for name in ("lat", "lon", "http_status", "n_levels_0_2000", "n_good_pres_0_2000",
                 "n_good_temp_0_2000", "n_good_psal_0_2000", "frac_good_temp_0_2000",
                 "frac_good_psal_0_2000", "max_pres_good"):
        qc[name] = pd.to_numeric(qc[name], errors="coerce")
    for name in ("POSITION_QC", "JULD_QC", "DATA_MODE"):
        qc[name] = qc[name].map(clean_token)
    for name in ("POSITION_QC", "JULD_QC"):
        qc[name] = pd.to_numeric(qc[name], errors="coerce").astype("Int64").astype(str)
    qc["error_clean"] = qc.error.astype(str).str.strip().replace(
        {"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan}
    )
    # Strict QC filter: successful download, position/time flags, delayed-mode data,
    # coverage thresholds over 0-2000 dbar, 34.5 S band, and 2003-2025 period.
    keep = (
        (qc.http_status == 200) & qc.error_clean.isna()
        & (qc.POSITION_QC == "1") & (qc.JULD_QC == "1") & (qc.DATA_MODE == "D")
        & (qc.n_levels_0_2000 >= 30) & (qc.n_good_pres_0_2000 >= 25)
        & (qc.n_good_temp_0_2000 >= 40) & (qc.n_good_psal_0_2000 >= 40)
        & (qc.frac_good_temp_0_2000 >= 0.60) & (qc.frac_good_psal_0_2000 >= 0.60)
        & (qc.max_pres_good >= 1800)
        & qc.date.between("2003-01-01", "2025-12-31")
        & qc.lat.between(-37.5, -31.5) & qc.lon.between(-55.0, 20.0)
    )
    result = qc.loc[keep].drop_duplicates(["wmo", "cyc"], keep="last").copy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(pd.read_csv(args.all_profiles, low_memory=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")   # staging file
    try:
        result.to_csv(temporary, index=False)
        temporary.replace(args.output)                                 # rename when fully written
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {args.output}: {len(result)} profiles")


if __name__ == "__main__":
    main()
