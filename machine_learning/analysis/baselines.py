"""Ridge and seasonal-climatology baselines on the eight blocked folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from machine_learning.dataset import build_datasets


FEATURES = ("x", "mask_padded", "lon", "lat", "days", "sst", "sss", "ssh", "obp", "season")
ALPHAS = np.logspace(-4, 5, 19)   # Ridge regularisation grid searched on the validation set


def features(dataset) -> np.ndarray:
    """Flatten every model-input tensor of a split into a single (n_samples, n_features) matrix."""
    matrix = np.concatenate([dataset._tensors[key].numpy().astype(np.float64).reshape(len(dataset), -1)
                             for key in FEATURES], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite baseline features")
    return matrix


def ridge_predictions(train, val, test) -> tuple[np.ndarray, float, float]:
    """Fit Ridge with per-fold α selection on validation; return test predictions and the chosen α."""
    x_train, x_val, x_test = (features(ds) for ds in (train, val, test))
    # Standardise features using train-only statistics.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train, x_val, x_test = ((matrix - mean) / std for matrix in (x_train, x_val, x_test))
    y_train = np.asarray(train.y_phys, dtype=float)
    y_val = np.asarray(val.y_phys, dtype=float)
    # Pick the α that minimises validation RMSE, then refit on the full training set.
    best_alpha, best_rmse = None, np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=float(alpha), fit_intercept=True).fit(x_train, y_train)
        rmse = float(np.sqrt(np.mean((model.predict(x_val) - y_val)**2)))
        if rmse < best_rmse:
            best_alpha, best_rmse = float(alpha), rmse
    model = Ridge(alpha=best_alpha, fit_intercept=True).fit(x_train, y_train)
    return model.predict(x_test), best_alpha, best_rmse


def climatology_predictions(train, test, regime: str) -> np.ndarray:
    """Return the train monthly-mean (M) or quarterly-mean (Q) target repeated on the test dates."""
    train_dates, test_dates = (pd.DatetimeIndex(ds.time_index) for ds in (train, test))
    labels_train = train_dates.month if regime == "M" else train_dates.quarter
    labels_test = test_dates.month if regime == "M" else test_dates.quarter
    seasonal = pd.DataFrame({"season": labels_train, "y": np.asarray(train.y_phys, dtype=float)})
    means = seasonal.groupby("season").y.mean().to_dict()
    fallback = float(np.mean(train.y_phys))                       # train mean when a season is missing in train
    return np.asarray([means.get(int(label), fallback) for label in labels_test], dtype=float)


def evaluate_fold(regime: str, fold: int) -> pd.DataFrame:
    """Run both baselines on one fold and return a long-format prediction table."""
    if regime not in ("M", "Q") or fold not in range(1, 9):
        raise ValueError("Expected regime M/Q and fold 1..8")
    train, val, test, _ = build_datasets(regime, fold, n_bins=20)
    ridge, alpha, val_rmse = ridge_predictions(train, val, test)
    climatology = climatology_predictions(train, test, regime)
    mode = "monthly_1" if regime == "M" else "quarter_mean"
    rows = []
    for method, predicted in (("ridge", ridge), ("climatology", climatology)):
        for sample, (index, date, truth, prediction) in enumerate(zip(
                test.global_time_indices, pd.DatetimeIndex(test.time_index), test.y_phys, predicted)):
            rows.append({"series_id": f"{method}_{regime}",
                         "label": f"{'Ridge' if method == 'ridge' else 'Climatology'} baseline {regime}",
                         "method_type": method, "regime": regime,
                         "surface_temporal_mode": mode, "fold": fold,
                         "sample_index_in_test": sample, "global_time_index": int(index),
                         "date": date, "y_true_phys": float(truth),
                         "y_pred_phys": float(prediction),
                         "ridge_best_alpha": alpha if method == "ridge" else np.nan,
                         "ridge_best_val_rmse": val_rmse if method == "ridge" else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("M", "Q", "both"), default="both")
    parser.add_argument("--fold", type=int, help="Run one fold for a focused check")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    regimes = ("M", "Q") if args.regime == "both" else (args.regime,)
    folds = (args.fold,) if args.fold is not None else range(1, 9)
    frames = []
    for regime in regimes:
        for fold in folds:
            print(f"Baseline {regime} fold {fold}", flush=True)
            frames.append(evaluate_fold(regime, fold))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    print(f"Wrote {len(output)} held-out baseline predictions")


if __name__ == "__main__":
    main()
