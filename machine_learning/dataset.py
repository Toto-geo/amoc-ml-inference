"""AMOC dataset builder: M monthly / Q quarter-mean splits with Argo rho and surface fields.

All paths are relative to the data-preparation tree. Wind stress is used to
build the target upstream, but is not a predictor in this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


DATA = Path(__file__).resolve().parents[1] / "data_preparation/data"
SCALES = {20: 20, 40: 10, 80: 5, 160: 3}      # n_bins -> n_lat_rows
SURFACES = {
    "sst": ("SST", "sst", "sst_bin_means"),
    "sss": ("SSS", "sss", "sss_bin_means"),
    "ssh": ("SSH", "ssh", "ssh_bin_means"),
    "obp": ("OBP", "virtual_wlt", "obp_bin_means"),
}
LON_MIN, LON_MAX = -51.5, 17.557667             # A-Z longitude range used to scale Argo coordinates
LAT_MIN, LAT_MAX = -37.5, -31.5                 # 3-degree latitude band


def fold_indices(length: int, fold: int) -> dict[str, np.ndarray]:
    """Return train/val/test index arrays for one of the eight blocked folds."""
    if length not in (816, 272) or fold not in range(1, 9):
        raise ValueError("Expected 816 monthly or 272 quarterly samples and fold 1..8")
    size = length // 8                                       # test-block length (102 months or 34 quarters)
    half = size // 2                                         # validation half-buffer on each side
    start = (fold - 1) * size
    test = np.arange(start, start + size)
    # Validation sits adjacent to the test block; edge folds use the opposite side.
    if fold == 1:
        val = np.arange(size, 2 * size)
    elif fold == 8:
        val = np.arange(6 * size, 7 * size)
    else:
        val = np.r_[np.arange(start - half, start), np.arange(start + size, start + size + half)]
    train = np.setdiff1d(np.arange(length), np.r_[test, val])
    return {"train": train, "val": val, "test": test}


def load_target(data: Path, regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the AMOC target in Sv and its monthly or quarterly timestamps."""
    period, name = ("monthly", "target_monthly") if regime == "M" else ("quarterly", "target_quarterly")
    path = data / "target" / period / f"oras5_{period}_target_34p5S_AZ_medianDepth_1958_2025.nc"
    with xr.open_dataset(path, engine="netcdf4") as ds:
        values = ds[name].values.astype(np.float32)
        time = ds.time_counter.values.astype("datetime64[ns]")
    expected = 816 if regime == "M" else 272
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {regime} target: {path}")
    return values, time


def load_surfaces(data: Path, bins: int, rows: int, regime: str, target_time: np.ndarray) -> dict[str, np.ndarray]:
    """Load SST/SSS/SSH/OBP at the requested bin count; aggregate to quarter means for Q."""
    result = {}
    for field, (folder, prefix, variable) in SURFACES.items():
        suffix = "" if rows == 20 else f"_lat{rows}"
        path = data / "input" / folder / f"oras5_{prefix}_input_AZ_{bins}bin{suffix}_means_1958_2025.nc"
        with xr.open_dataset(path, engine="netcdf4") as ds:
            values = ds[variable].values.astype(np.float32)   # (816, bins)
            time = ds.time_counter.values.astype("datetime64[ns]")
            if int(ds.attrs["surface_lat_rows"]) != rows:
                raise ValueError(f"Wrong latitude-row geometry: {path}")
        if values.shape != (816, bins) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} values: {path}")
        if regime == "M":
            if not np.array_equal(time, target_time):
                raise ValueError(f"Monthly time mismatch: {path}")
            result[field] = values
        else:
            # Quarterly regime: average every three consecutive months, timestamped at month 3.
            month = time.astype("datetime64[M]")
            if not np.all(np.diff(month.astype(np.int64)) == 1):
                raise ValueError(f"Surface months must be consecutive: {path}")
            if not np.array_equal(month.reshape(272, 3)[:, 2], target_time.astype("datetime64[M]")):
                raise ValueError(f"Quarterly month alignment mismatch: {path}")
            result[field] = values.reshape(272, 3, bins).mean(axis=1).astype(np.float32)
    return result


def load_argo(data: Path, regime: str, target_time: np.ndarray) -> dict[str, np.ndarray]:
    """Load virtual-Argo rho, position, mask and relative time for the chosen regime."""
    suffix = "Q_" if regime == "Q" else ""
    dim = "sample" if regime == "Q" else "time"
    path = data / "input/virtual_argo" / f"oras5_virtual_argo_density_input_{suffix}34p5S_pm3deg_1958_2025.nc"
    with xr.open_dataset(path, engine="netcdf4") as ds:
        mask = ds.profile_mask.values.astype(bool)                 # (T, P_native) valid-profile flag
        rho = ds.rho.values.astype(np.float32)                     # (T, P_native, Z)
        lon, lat = ds.lon.values.astype(np.float32), ds.lat.values.astype(np.float32)
        time = ds[dim].values.astype("datetime64[ns]")
        month_in_quarter = ds.month_in_quarter.values.astype(np.int8) if regime == "Q" else None
        labels = ds.quarter_label.values.astype(str) if regime == "Q" else None
        z = ds.z.values.astype(np.float32)
        pressure = ds.pressure.values.astype(np.float32)
    if rho.shape[:2] != mask.shape or lon.shape != mask.shape or lat.shape != mask.shape:
        raise ValueError(f"Argo shape mismatch: {path}")
    if regime == "M":
        if not np.array_equal(time, target_time):
            raise ValueError("Monthly Argo and target times differ")
        days = np.full(mask.shape, np.float32(0.5), dtype=np.float32)   # mid-month position in [0, 1]
    else:
        expected_labels = np.array([f"{int(t.astype('datetime64[Y]').astype(int) + 1970)}-Q{int(t.astype('datetime64[M]').astype(int) % 12 // 3 + 1)}"
                                    for t in target_time], dtype=str)
        if not np.array_equal(labels, expected_labels):
            raise ValueError("Quarterly Argo and target quarter labels differ")
        if not np.array_equal(time.astype("datetime64[M]") + np.timedelta64(1, "M"),
                              target_time.astype("datetime64[M]")):
            raise ValueError("Quarterly Argo must precede target by one month")
        if np.any(mask & ((month_in_quarter < 0) | (month_in_quarter > 2))):
            raise ValueError("Quarterly Argo month_in_quarter must be 0, 1 or 2")
        days = (month_in_quarter.astype(np.float32) + np.float32(0.5)) / np.float32(3.0)  # mid-month of each sub-month in [0, 1]
    if np.any(mask & (~np.isfinite(lon) | ~np.isfinite(lat))):
        raise ValueError("Valid Argo profiles need finite coordinates")
    return {"rho": rho, "profile_mask": mask, "lon": lon, "lat": lat,
            "days": days, "z": z, "pressure": pressure}


def normalize(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise values using the train subset mean and std (no leakage from val/test)."""
    physical = values.astype(np.float32)
    mean = physical[train].astype(np.float64).mean(axis=0, dtype=np.float64)
    std = physical[train].astype(np.float64).std(axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid train-only normalization statistics")
    return ((physical.astype(np.float64) - mean) / std).astype(np.float32), mean, std


def pad_argo(argo: dict[str, np.ndarray], train: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Normalise rho on train-only stats and pack each sample's profiles into a fixed-width tensor."""
    mask, rho = argo["profile_mask"], argo["rho"]
    # Train-only normalisation of rho, computed only on valid training profiles.
    train_mask = mask & np.isin(np.arange(len(mask)), train)[:, None]
    train_rho = np.where(train_mask[..., None], rho, np.nan)
    mean = np.nanmean(train_rho, axis=(0, 1), dtype=np.float64)        # per-depth mean (Z,)
    std = np.nanstd(train_rho, axis=(0, 1), ddof=0, dtype=np.float64)  # per-depth std (Z,)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid Argo training statistics")
    normalized = ((rho.astype(np.float64) - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    normalized = np.where(np.isfinite(normalized), normalized, np.float32(0)).astype(np.float32)
    # Scale coordinates and relative time to [0, 1] on the valid positions only.
    lon_scaled = np.where(mask, (argo["lon"] - LON_MIN) / (LON_MAX - LON_MIN), np.nan).astype(np.float32)
    lat_scaled = np.where(mask, (argo["lat"] - LAT_MIN) / (LAT_MAX - LAT_MIN), np.nan).astype(np.float32)
    days_scaled = np.where(mask, argo["days"], np.nan).astype(np.float32)
    counts = mask.sum(axis=1).astype(np.int64)
    width = int(counts.max())                                           # P: fixed per-sample profile capacity
    T, Z = rho.shape[0], rho.shape[2]
    # Pack per-sample profiles into a dense (T, P, ...) tensor and set the padding mask.
    padded = {"x": np.zeros((T, width, Z), dtype=np.float32),
              **{name: np.zeros((T, width), dtype=np.float32) for name in ("mask_padded", "lon", "lat", "days")}}
    for t in range(T):
        indices = np.where(mask[t])[0]
        n = len(indices)
        padded["x"][t, :n] = normalized[t, indices]
        padded["mask_padded"][t, :n] = 1
        padded["lon"][t, :n] = lon_scaled[t, indices]
        padded["lat"][t, :n] = lat_scaled[t, indices]
        padded["days"][t, :n] = days_scaled[t, indices]
    padded.update(lon_scaled=lon_scaled, lat_scaled=lat_scaled, days_scaled=days_scaled,
                  profile_mask=mask, counts_per_sample=counts)
    return padded, {"argo_norm_means": {"rho": mean.astype(np.float32)},
                    "argo_norm_stds": {"rho": std.astype(np.float32)},
                    "P": width, "F": Z, "Z": Z, "native_pos": mask.shape[1],
                    "argo_z_axis": argo["z"], "argo_pressure_axis": argo["pressure"]}


def season(time: np.ndarray, regime: str) -> np.ndarray:
    """Return (sin, cos) of the month-of-year (M) or quarter-of-year (Q) angle."""
    months = time.astype("datetime64[M]").astype(np.int64) % 12
    angle = (2.0 * np.pi) * (months if regime == "M" else months // 3) / (12 if regime == "M" else 4)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


@dataclass
class Split:
    name: str
    arrays: dict[str, np.ndarray]
    global_time_indices: np.ndarray
    time_index: np.ndarray
    y_phys: np.ndarray
    lon_scaled: np.ndarray
    lat_scaled: np.ndarray
    days_scaled: np.ndarray
    profile_mask: np.ndarray
    counts_per_sample: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.global_time_indices)


class ModelDataset(Dataset):
    """PyTorch view of one split; the model has no wind-stress slot."""

    def __init__(self, split: Split, stats: dict):
        self.split = split
        self.norm_stats = stats
        self.global_time_indices = split.global_time_indices
        self.time_index = split.time_index
        self.y_phys = split.y_phys
        self.lon_scaled = split.lon_scaled
        self.lat_scaled = split.lat_scaled
        self.days_scaled = split.days_scaled
        self.profile_mask = split.profile_mask
        self.counts_per_sample = split.counts_per_sample
        self.split_name = split.name
        self.T = split.n_samples
        self.P = stats["P"]
        self.F = stats["F"]
        self.native_pos = stats["native_pos"]
        self.surface_feature_dim = 4 * stats["surface_n_bins"]
        self.season_feature_dim = 2
        self._tensors = {name: torch.tensor(values, dtype=torch.float32)
                         for name, values in split.arrays.items()}
        if any(not torch.isfinite(tensor).all() for tensor in self._tensors.values()):
            raise ValueError(f"Non-finite model input in {split.name} split")

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: values[index] for name, values in self._tensors.items()}


def build_datasets(regime: str, fold: int, *, n_bins: int = 20,
                   data_root: Path = DATA) -> tuple[ModelDataset, ModelDataset, ModelDataset, dict]:
    """Return train/validation/test datasets for one fold, along with normalisation statistics."""
    if regime not in ("M", "Q") or n_bins not in SCALES:
        raise ValueError("Use regime M/Q and n_bins 20/40/80/160")
    if not Path(data_root).exists():
        raise SystemExit(
            f"Prepared data not found at '{data_root}'. Extract data.zip from the Zenodo record "
            "(https://doi.org/10.5281/zenodo.23287700) into the repository root, or rebuild it "
            "following data_preparation/README.md, before training or evaluating."
        )
    rows = SCALES[n_bins]
    y_phys, time = load_target(data_root, regime)
    indices = fold_indices(len(y_phys), fold)
    # Train-only target normalisation; test values are standardised with the same stats.
    y_mean = float(np.mean(y_phys[indices["train"]], dtype=np.float64))
    y_std = float(np.std(y_phys[indices["train"]], ddof=0, dtype=np.float64))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("Invalid train-only target standard deviation")
    y = ((y_phys.astype(np.float64) - y_mean) / y_std).astype(np.float32)
    surfaces = load_surfaces(data_root, n_bins, rows, regime, time)
    surface_norm, surface_mean, surface_std = {}, {}, {}
    for name, values in surfaces.items():
        surface_norm[name], surface_mean[name], surface_std[name] = normalize(values, indices["train"])
    argo = load_argo(data_root, regime, time)
    padded, argo_stats = pad_argo(argo, indices["train"])
    seasonal = season(time, regime)
    stats = {"regime": regime, "fold_number": fold,
             "surface_temporal_mode": "monthly_1" if regime == "M" else "quarter_mean",
             "surface_n_bins": n_bins, "surface_lat_rows": rows,
             "surface_feature_dims": {name: n_bins for name in SURFACES},
             "surface_months_per_sample": 1, "season_feature_dim": 2,
             "y_mean_train": y_mean, "y_std_train": y_std,
             "surface_norm_means": surface_mean, "surface_norm_stds": surface_std,
             **argo_stats}
    # Materialise each split as a ModelDataset backed by its own per-sample tensors.
    splits = []
    for name, idx in indices.items():
        arrays = {key: padded[key][idx].copy() for key in ("x", "mask_padded", "lon", "lat", "days")}
        arrays.update({key: surface_norm[key][idx].copy() for key in SURFACES})
        arrays.update(season=seasonal[idx].copy(), y=y[idx].copy())
        split = Split(name, arrays, idx.copy(), time[idx].copy(), y_phys[idx].copy(),
                      padded["lon_scaled"][idx].copy(), padded["lat_scaled"][idx].copy(),
                      padded["days_scaled"][idx].copy(), padded["profile_mask"][idx].copy(),
                      padded["counts_per_sample"][idx].copy())
        splits.append(ModelDataset(split, stats))
    return *splits, stats
