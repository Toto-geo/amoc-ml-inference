"""Generate deterministic synthetic inputs for the minimal end-to-end test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated/data"
N_MONTHS, N_QUARTERS = 816, 272
N_PROFILES, N_DEPTHS, N_BINS = 32, 54, 20
FIELDS = {
    "SST": ("sst", "sst_bin_means"),
    "SSS": ("sss", "sss_bin_means"),
    "SSH": ("ssh", "ssh_bin_means"),
    "OBP": ("virtual_wlt", "obp_bin_means"),
}
SYNTHETIC = {"scientific_use": "Synthetic smoke test only"}


def _write(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, value in dataset.data_vars.items()
        if value.ndim and value.dtype.kind in "fiu"
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def generate(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write monthly and quarterly fixtures without replacing existing data."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing test data: {output}")

    rng = np.random.default_rng(20261014)
    months = np.arange("1958-01", "2026-01", dtype="datetime64[M]").astype("datetime64[ns]")
    quarter_target = months.reshape(N_QUARTERS, 3)[:, 2]
    phase = np.arange(N_MONTHS, dtype=np.float32)
    target = (12 + 2 * np.sin(2 * np.pi * phase / 12)
              + 0.5 * np.sin(2 * np.pi * phase / 67)).astype(np.float32)

    _write(
        xr.Dataset({"target_monthly": (("time_counter",), target)},
                   coords={"time_counter": months}, attrs=SYNTHETIC),
        output / "target/monthly/oras5_monthly_target_34p5S_AZ_medianDepth_1958_2025.nc",
    )
    quarterly_target = target.reshape(N_QUARTERS, 3).mean(1).astype(np.float32)
    _write(
        xr.Dataset({"target_quarterly": (("time_counter",), quarterly_target)},
                   coords={"time_counter": quarter_target}, attrs=SYNTHETIC),
        output / "target/quarterly/oras5_quarterly_target_34p5S_AZ_medianDepth_1958_2025.nc",
    )

    bins = np.arange(N_BINS, dtype=np.int32)
    for offset, (folder, (prefix, variable)) in enumerate(FIELDS.items()):
        values = (np.sin(phase[:, None] / (8 + offset) + bins[None, :] / 5)
                  + 0.03 * rng.standard_normal((N_MONTHS, N_BINS))).astype(np.float32)
        dataset = xr.Dataset(
            {variable: (("time_counter", "bin"), values)},
            coords={"time_counter": months, "bin": bins},
            attrs={**SYNTHETIC, "surface_lat_rows": 20},
        )
        _write(dataset, output / f"input/{folder}/oras5_{prefix}_input_AZ_20bin_means_1958_2025.nc")

    z = np.arange(N_DEPTHS, dtype=np.float32) * 20
    positions = np.arange(N_PROFILES, dtype=np.float32)
    lon = np.broadcast_to(np.linspace(-51, 17, N_PROFILES, dtype=np.float32),
                          (N_MONTHS, N_PROFILES)).copy()
    lat = np.broadcast_to((-34.5 + 2 * np.sin(positions))[None, :],
                          (N_MONTHS, N_PROFILES)).copy().astype(np.float32)
    rho = (1025 + 0.018 * z[None, None, :]
           + 0.08 * np.sin(phase[:, None, None] / 12 + positions[None, :, None] / 4)
           + 0.003 * rng.standard_normal((N_MONTHS, N_PROFILES, N_DEPTHS))).astype(np.float32)
    mask_m = np.ones((N_MONTHS, N_PROFILES), dtype=np.int8)
    monthly = xr.Dataset(
        {"rho": (("time", "pos", "z"), rho),
         "profile_mask": (("time", "pos"), mask_m),
         "lon": (("time", "pos"), lon),
         "lat": (("time", "pos"), lat),
         "pressure": (("z",), z.copy())},
        coords={"time": months, "pos": np.arange(N_PROFILES), "z": z},
        attrs=SYNTHETIC,
    )
    _write(monthly, output / "input/virtual_argo/oras5_virtual_argo_density_input_34p5S_pm3deg_1958_2025.nc")

    quarter_sample = quarter_target.astype("datetime64[M]") - np.timedelta64(1, "M")
    quarter_labels = np.array([
        f"{int(t.astype('datetime64[Y]').astype(int) + 1970)}-"
        f"Q{int(t.astype('datetime64[M]').astype(int) % 12 // 3 + 1)}"
        for t in quarter_target
    ], dtype=str)
    quarterly = xr.Dataset(
        {"rho": (("sample", "pos", "z"),
                 rho.reshape(N_QUARTERS, 3, N_PROFILES, N_DEPTHS).mean(1)),
         "profile_mask": (("sample", "pos"),
                          np.ones((N_QUARTERS, N_PROFILES), dtype=np.int8)),
         "lon": (("sample", "pos"), lon.reshape(N_QUARTERS, 3, N_PROFILES).mean(1)),
         "lat": (("sample", "pos"), lat.reshape(N_QUARTERS, 3, N_PROFILES).mean(1)),
         "month_in_quarter": (("sample", "pos"),
                              np.broadcast_to(np.arange(N_PROFILES) % 3,
                                              (N_QUARTERS, N_PROFILES)).astype(np.int8)),
         "quarter_label": (("sample",), quarter_labels),
         "pressure": (("z",), z.copy())},
        coords={"sample": quarter_sample.astype("datetime64[ns]"),
                "pos": np.arange(N_PROFILES), "z": z},
        attrs=SYNTHETIC,
    )
    _write(
        quarterly,
        output / "input/virtual_argo/oras5_virtual_argo_density_input_Q_34p5S_pm3deg_1958_2025.nc",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        output = generate(args.output_dir)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    print(f"Wrote synthetic test data to {output}")


if __name__ == "__main__":
    main()
