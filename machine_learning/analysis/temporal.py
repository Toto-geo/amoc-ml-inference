"""Post-hoc seasonal and frequency-band diagnostics on reconstructed series.

This full-record decomposition is descriptive; it must not be used as a
training feature or as the train-only seasonal-climatology baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import chi2

from machine_learning.dataset import fold_indices


WINDOW = {"M": 13, "Q": 5}       # centered rolling-mean width per regime (months / quarters)


def decompose(values: np.ndarray, dates: pd.DatetimeIndex, regime: str) -> pd.DataFrame:
    """Split a series into seasonal cycle, deseasonalised anomaly, low-pass and high-frequency bands."""
    if regime not in WINDOW or len(values) != len(dates) or len(values) == 0:
        raise ValueError("Invalid temporal series")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Temporal dates must be unique and ordered")
    original = pd.Series(np.asarray(values, dtype=float), index=dates)
    if not np.isfinite(original).all():
        raise ValueError("Temporal values must be finite")
    # Seasonal cycle: group-mean by month (M) or by quarter (Q), broadcast back to the dates.
    season = pd.Series(dates.month if regime == "M" else dates.quarter, index=dates)
    seasonal = original.groupby(season).transform("mean")
    anomaly = original - seasonal
    low = anomaly.rolling(window=WINDOW[regime], center=True, min_periods=1).mean()    # low-pass band
    high = anomaly - low                                                               # high-frequency residual
    return pd.DataFrame({"seasonal_cycle": seasonal, "deseasonalized_anomaly": anomaly,
                         "low_pass": low, "high_frequency": high}, index=dates)


def correlation(true: pd.Series, predicted: pd.Series) -> float:
    return float(np.corrcoef(true.to_numpy(dtype=float), predicted.to_numpy(dtype=float))[0, 1])


def analyze(seed_mean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decompose truth and seed-mean prediction and compute per-band Pearson correlations."""
    required = {"regime", "bins", "fold", "global_time_index", "date", "y_true_phys", "y_pred_phys"}
    if required - set(seed_mean.columns) or seed_mean.empty:
        raise ValueError("Expected complete seed-mean paper predictions")
    components, fold_rows, summaries = [], [], []
    for (regime, bins), frame in seed_mean.groupby(["regime", "bins"]):
        expected_size = 816 if regime == "M" else 272
        frame = frame.sort_values("global_time_index").reset_index(drop=True).copy()
        if regime not in WINDOW or int(bins) != 20 or not np.array_equal(
                frame.global_time_index.to_numpy(), np.arange(expected_size)):
            raise ValueError("Expected complete 20-bin monthly or quarterly series")
        if any(not np.array_equal(frame.loc[frame.fold.eq(fold), "global_time_index"].to_numpy(),
                                  fold_indices(expected_size, fold)["test"]) for fold in range(1, 9)):
            raise ValueError("Temporal rows disagree with the blocked folds")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
        regression_slope = float(np.polyfit(frame.y_true_phys, frame.y_pred_phys, deg=1)[0])
        # Decompose both truth and prediction once; store every component as a new column.
        for label, column in (("true", "y_true_phys"), ("pred", "y_pred_phys")):
            part = decompose(frame[column].to_numpy(), dates, regime)
            for component in part.columns:
                frame[f"{label}_{component}"] = part[component].to_numpy()
        # Per-band Pearson correlation on the full record and on each blocked fold.
        for component in ("low_pass", "high_frequency"):
            true, pred = frame[f"true_{component}"], frame[f"pred_{component}"]
            for fold, part in frame.groupby("fold"):
                fold_rows.append({"regime": regime, "bins": bins, "component": component,
                                  "fold": int(fold), "pearson": correlation(
                                      part[f"true_{component}"], part[f"pred_{component}"])})
            values = [row["pearson"] for row in fold_rows if row["regime"] == regime
                      and row["bins"] == bins and row["component"] == component]
            summaries.append({"regime": regime, "bins": bins, "component": component,
                              "pearson_full_record": correlation(true, pred),
                              "pearson_fold_mean": float(np.mean(values)),
                              "pearson_fold_std": float(np.std(values, ddof=1)),
                              "reconstruction_slope": regression_slope})
        components.append(frame)
    return (pd.concat(components, ignore_index=True), pd.DataFrame(fold_rows),
            pd.DataFrame(summaries))


def target_structure(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ACF, Ljung-Box and Welch spectrum of the monthly target's high-frequency band."""
    monthly = components.loc[components.regime.eq("M") & components.bins.eq(20)]
    monthly = monthly.sort_values("global_time_index")
    if len(monthly) != 816:
        raise ValueError("Expected the complete monthly target")
    high = monthly.true_high_frequency.to_numpy(dtype=float)[6:-6]      # drop edges affected by the 13-month roll
    n = len(high)
    centered = high - high.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        raise ValueError("Monthly high-frequency target has zero variance")
    # Autocorrelation up to lag 12 months with the classical 95% confidence bound.
    acf = np.array([1.0] + [float(np.dot(centered[lag:], centered[:-lag]) / denom)
                            for lag in range(1, 13)])
    bound = 1.96 / np.sqrt(n)
    acf_table = pd.DataFrame({"lag_months": np.arange(1, 13), "acf": acf[1:],
                              "n_samples": n, "acf_95_bound_lower": -bound,
                              "acf_95_bound_upper": bound})
    # Ljung-Box test at lags 6 and 12.
    lb_rows = []
    for lag in (6, 12):
        q = n * (n + 2) * sum(acf[k]**2 / (n - k) for k in range(1, lag + 1))
        lb_rows.append({"lag_months": lag, "n_samples": n, "q_stat": float(q),
                        "p_value": float(chi2.sf(q, df=lag))})
    # Welch power spectrum with a Hann window and 50% overlap.
    frequencies, psd = signal.welch(high, fs=1.0, window="hann", nperseg=128,
                                     noverlap=64, detrend="constant", scaling="density")
    with np.errstate(divide="ignore"):
        periods = 1.0 / frequencies
    spectrum = pd.DataFrame({"frequency_cycles_per_month": frequencies,
                             "period_months": periods, "psd": psd})
    positive = spectrum.loc[spectrum.frequency_cycles_per_month.gt(0)]
    total = float(np.trapezoid(positive.psd, positive.frequency_cycles_per_month))
    # Fraction of PSD in four period bands (longest to shortest).
    bands = (("period_gt_12_months", 0.0, 1 / 12),
             ("period_6_to_12_months", 1 / 12, 1 / 6),
             ("period_3_to_6_months", 1 / 6, 1 / 3),
             ("period_2_to_3_months", 1 / 3, 0.5 + 1e-12))
    power_rows = []
    for label, low, high_frequency in bands:
        part = positive.loc[positive.frequency_cycles_per_month.ge(low)
                            & positive.frequency_cycles_per_month.lt(high_frequency)]
        power = float(np.trapezoid(part.psd, part.frequency_cycles_per_month)) if len(part) > 1 else 0.0
        power_rows.append({"band": label, "power": power, "power_fraction": power / total})
    return acf_table, pd.DataFrame(lb_rows), spectrum, pd.DataFrame(power_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="seed_mean_predictions.csv from machine_learning.evaluate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    components, folds, summary = analyze(pd.read_csv(args.predictions))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    components.to_csv(args.output_dir / "temporal_components.csv", index=False)
    folds.to_csv(args.output_dir / "band_skill_by_fold.csv", index=False)
    summary.to_csv(args.output_dir / "band_skill_summary.csv", index=False)
    if "M" in set(components.regime):
        acf, ljung_box, spectrum, power = target_structure(components)
        acf.to_csv(args.output_dir / "high_frequency_acf.csv", index=False)
        ljung_box.to_csv(args.output_dir / "high_frequency_ljung_box.csv", index=False)
        spectrum.to_csv(args.output_dir / "high_frequency_psd.csv", index=False)
        power.to_csv(args.output_dir / "high_frequency_band_power.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
