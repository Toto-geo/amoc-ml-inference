"""Export checkpoint predictions and aggregate the out-of-fold skill.

Predictions are averaged across seeds at each date; metrics are computed on
each held-out fold; the eight fold metrics are then summarised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import DATA, SCALES, build_datasets, fold_indices
from .model import build_model, load_checkpoint
from .train import SEEDS, evaluate


METRICS = ("rmse", "mae", "pearson", "r2", "spread_ratio", "bias")
PREDICTION_COLUMNS = ("regime", "bins", "fold", "seed", "global_time_index",
                      "date", "y_true_phys", "y_pred_phys")


def export_predictions(checkpoint_path: Path, output_dir: Path, *, regime: str,
                       bins: int, fold: int, seed: int,
                       data_root: Path = DATA) -> dict:
    """Load one checkpoint, write its test-set predictions and metrics into output_dir."""
    if regime not in ("M", "Q") or bins not in SCALES or fold not in range(1, 9) or seed not in SEEDS:
        raise ValueError("Expected a paper regime, resolution, fold and seed")
    train, _, test, stats = build_datasets(
        regime, fold, n_bins=bins, data_root=data_root
    )
    model = build_model(train, initialize_assignment=False)
    checkpoint = load_checkpoint(model, checkpoint_path)
    if int(checkpoint["config"]["seed"]) != seed:
        raise ValueError("Checkpoint seed differs from the requested seed")

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(test, batch_size=16, shuffle=False):
            prediction, _ = model(batch)
            predictions.append(prediction.cpu())
    predicted = torch.cat(predictions).numpy()                  # normalised predictions
    true = test._tensors["y"].numpy()                           # normalised targets
    if len(predicted) != len(test) or not np.isfinite(predicted).all():
        raise ValueError("Invalid checkpoint predictions")
    frame = pd.DataFrame({
        "regime": regime, "bins": bins, "fold": fold, "seed": seed,
        "global_time_index": test.global_time_indices,
        "date": np.datetime_as_string(test.time_index, unit="D"),
        "y_true_norm": true, "y_pred_norm": predicted,
        # Physical-space columns: undo the train-only normalisation.
        "y_true_phys": true.astype(np.float64) * stats["y_std_train"] + stats["y_mean_train"],
        "y_pred_phys": predicted.astype(np.float64) * stats["y_std_train"] + stats["y_mean_train"],
    })
    metrics = evaluate(model, DataLoader(test, batch_size=16), stats, torch.device("cpu"))
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    result = {"regime": regime, "bins": bins, "fold": fold, "seed": seed,
              "n_test": len(test), "checkpoint": str(checkpoint_path), **metrics}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def skill(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, Pearson, R2, spread ratio and bias for a 1-D prediction vector."""
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 1 or len(true) < 2:
        raise ValueError("Expected equally sized nonempty prediction vectors")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Predictions and targets must be finite")
    error = pred - true
    true_std, pred_std = true.std(ddof=0), pred.std(ddof=0)
    if true_std <= 0 or pred_std <= 0:
        raise ValueError("Fold target and prediction must have nonzero spread")
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": float(np.corrcoef(true, pred)[0, 1]),
        "r2": float(1 - np.sum(error**2) / np.sum((true - true.mean())**2)),
        "spread_ratio": float(pred_std / true_std),
        "bias": float(np.mean(error)),
    }


def aggregate_predictions(frame: pd.DataFrame, *, expected_folds: int = 8,
                          expected_seeds: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average predictions across seeds, then compute per-fold skill and a per-regime summary."""
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing prediction columns or rows: {sorted(missing)}")
    frame = frame.copy()
    keys = ["regime", "bins", "fold", "seed", "global_time_index"]
    if frame.duplicated(keys).any() or frame[list(PREDICTION_COLUMNS[6:])].isna().any().any():
        raise ValueError("Duplicate or missing test predictions")
    # Check that every (regime, bins) group covers all expected folds and seeds.
    for (regime, bins), group in frame.groupby(["regime", "bins"]):
        if regime not in ("M", "Q") or int(bins) not in SCALES:
            raise ValueError("Unexpected paper regime or resolution")
        folds = sorted(group.fold.unique())
        if len(folds) != expected_folds or folds != list(range(1, expected_folds + 1)):
            raise ValueError(f"Incomplete fold coverage for {regime}/{bins}")
        seeds_by_fold = [set(part.seed) for _, part in group.groupby("fold")]
        if any(len(seeds) != expected_seeds or seeds != seeds_by_fold[0]
               for seeds in seeds_by_fold):
            raise ValueError(f"Incomplete or inconsistent seed coverage for {regime}/{bins}")
        for (fold, seed), run in group.groupby(["fold", "seed"]):
            expected = fold_indices(816 if regime == "M" else 272, int(fold))["test"]
            if not np.array_equal(np.sort(run.global_time_index.to_numpy()), expected):
                raise ValueError(f"Incomplete test block for {regime}/{bins}/fold {fold}/seed {seed}")

    # Each (regime, bins, fold, sample) must have one date and one truth across all seeds.
    sample = ["regime", "bins", "fold", "global_time_index"]
    consistency = frame.groupby(sample).agg(date_count=("date", "nunique"),
                                            truth_count=("y_true_phys", "nunique"))
    if (consistency.date_count != 1).any() or (consistency.truth_count != 1).any():
        raise ValueError("Seeds disagree on a test date or target")
    # Seed-mean prediction per sample, then per-fold skill on those seed means.
    mean = (frame.groupby(sample, as_index=False)
            .agg(date=("date", "first"), y_true_phys=("y_true_phys", "first"),
                 y_pred_phys=("y_pred_phys", "mean"))
            .sort_values(["regime", "bins", "global_time_index"]).reset_index(drop=True))
    rows = []
    for (regime, bins, fold), part in mean.groupby(["regime", "bins", "fold"]):
        rows.append({"regime": regime, "bins": bins, "fold": fold,
                     **skill(part.y_true_phys.to_numpy(), part.y_pred_phys.to_numpy())})
    folds = pd.DataFrame(rows).sort_values(["regime", "bins", "fold"]).reset_index(drop=True)
    # Fold-level mean/std per metric: the final publishable summary.
    summary = (folds.groupby(["regime", "bins"], as_index=False)
               .agg(**{f"{metric}_{stat}": (metric, stat)
                       for metric in METRICS for stat in ("mean", "std")}))
    return mean, folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Evaluate one trusted checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--regime", choices=("M", "Q"), required=True)
    export.add_argument("--bins", type=int, choices=tuple(SCALES), required=True)
    export.add_argument("--fold", type=int, required=True)
    export.add_argument("--seed", type=int, required=True)
    export.add_argument("--data-root", type=Path, default=DATA)
    aggregate = commands.add_parser("aggregate", help="Seed mean, then fold skill")
    aggregate.add_argument("predictions", type=Path, nargs="+")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--expected-folds", type=int, default=8)
    aggregate.add_argument("--expected-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.command == "export":
        result = export_predictions(args.checkpoint, args.output_dir, regime=args.regime,
                                    bins=args.bins, fold=args.fold, seed=args.seed,
                                    data_root=args.data_root)
        print(json.dumps(result, indent=2))
    else:
        frames = [pd.read_csv(path) for path in args.predictions]
        mean, folds, summary = aggregate_predictions(
            pd.concat(frames, ignore_index=True), expected_folds=args.expected_folds,
            expected_seeds=args.expected_seeds)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        mean.to_csv(args.output_dir / "seed_mean_predictions.csv", index=False)
        folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
