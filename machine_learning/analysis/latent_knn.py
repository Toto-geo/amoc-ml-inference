"""M-regime pre-head k-NN diagnostic computed with distances within each seed only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from machine_learning.dataset import build_datasets
from machine_learning.model import build_model, load_checkpoint
from .temporal import decompose
from machine_learning.train import SEEDS


K_VALUES = (5, 10, 20)       # neighbour counts to probe
BUFFER_MONTHS = 13           # exclude samples within this many months of the query
COMPONENTS = ("full", "low_pass", "high_frequency")


def export_embeddings(campaign_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Read every M/20 run and return (160 x 102, 1120) pre-head embeddings with their metadata.

    Each seed defines its own latent space, so seed-specific coordinates must never be averaged.
    """
    metadata, arrays = [], []
    for seed in SEEDS:
        for fold in range(1, 9):
            checkpoint_path = campaign_root / "M_20" / f"fold_{fold:02d}" / f"seed_{seed:05d}" / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            train, _, test, stats = build_datasets("M", fold, n_bins=20)
            model = build_model(train, initialize_assignment=False)
            checkpoint = load_checkpoint(model, checkpoint_path)
            if int(checkpoint["config"]["seed"]) != seed:
                raise ValueError(f"Checkpoint seed mismatch: {checkpoint_path}")
            model.eval()
            embeddings, predictions = [], []
            with torch.inference_mode():
                for batch in DataLoader(test, batch_size=16, shuffle=False):
                    prediction, (_, embedding) = model(batch)
                    embeddings.append(embedding.cpu().numpy().astype(np.float32))
                    predictions.append(prediction.cpu().numpy())
            array = np.concatenate(embeddings)                       # (n_test, 1120) pre-head embedding
            pred = np.concatenate(predictions).astype(np.float64)
            true = test._tensors["y"].numpy().astype(np.float64)
            if array.shape != (102, 1120) or not np.isfinite(array).all():
                raise ValueError(f"Unexpected embedding shape or values: {checkpoint_path}")
            arrays.append(array)
            metadata.append(pd.DataFrame({
                "fold": fold, "seed": seed, "global_time_index": test.global_time_indices,
                "date": np.datetime_as_string(test.time_index, unit="D"),
                "y_true_phys": true * stats["y_std_train"] + stats["y_mean_train"],
                "y_pred_phys": pred * stats["y_std_train"] + stats["y_mean_train"],
            }))
    return np.concatenate(arrays), pd.concat(metadata, ignore_index=True)


def load_embeddings(npz_path: Path, metadata_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pre-computed pre-head embeddings and their matching metadata table."""
    with np.load(npz_path, allow_pickle=False) as payload:
        embeddings = payload["embeddings"].astype(np.float32, copy=False)
    metadata = pd.read_csv(metadata_path, parse_dates=["date"])
    required = {"seed", "fold", "global_time_index", "date", "y_true_phys", "y_pred_phys"}
    if required - set(metadata) or embeddings.shape != (len(metadata), 1120):
        raise ValueError("Incompatible pre-head embeddings and metadata")
    return embeddings, metadata


def _components(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the full series and its low-pass / high-frequency decomposition for truth and prediction."""
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    true = frame.y_true_phys.to_numpy(dtype=float)
    pred = frame.y_pred_phys.to_numpy(dtype=float)
    true_parts = decompose(true, dates, "M")
    pred_parts = decompose(pred, dates, "M")
    return {"full": (true, pred),
            "low_pass": (true_parts.low_pass.to_numpy(), pred_parts.low_pass.to_numpy()),
            "high_frequency": (true_parts.high_frequency.to_numpy(),
                               pred_parts.high_frequency.to_numpy())}


def nearest_neighbors(coordinates: np.ndarray, time: np.ndarray) -> dict[int, np.ndarray]:
    """Return the k nearest-neighbour indices for each K, excluding samples within BUFFER_MONTHS."""
    distances = pairwise_distances(coordinates, metric="euclidean")
    # Mask out temporally close samples so neighbours cannot leak information from the query's window.
    distances[np.abs(time[:, None] - time[None, :]) <= BUFFER_MONTHS] = np.inf
    neighbors = {}
    for k in K_VALUES:
        partial = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        neighbor_distances = np.take_along_axis(distances, partial, axis=1)
        order = np.argsort(neighbor_distances, axis=1)
        neighbors[k] = np.take_along_axis(partial, order, axis=1)
        if not np.isfinite(neighbor_distances).all():
            raise ValueError("Too few temporally separated neighbors")
    return neighbors


def analyze(embeddings: np.ndarray, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-seed nearest-neighbour floor RMSE and model RMSE on each frequency component."""
    if embeddings.shape != (20 * 816, 1120) or len(metadata) != len(embeddings):
        raise ValueError("Expected 20 complete seed-specific M embeddings")
    seeds = sorted(metadata.seed.astype(int).unique())
    if set(seeds) != set(SEEDS):
        raise ValueError("Expected the paper's 20 seeds")
    floor_rows, model_rows = [], []
    for seed in seeds:
        # Sort this seed's samples by their global month index so neighbours can be found by time.
        mask = metadata.seed.astype(int).to_numpy() == seed
        frame = metadata.loc[mask].copy()
        order = np.argsort(frame.global_time_index.to_numpy(dtype=np.int64))
        frame = frame.iloc[order].reset_index(drop=True)
        coordinates = embeddings[mask][order]
        if not np.array_equal(frame.global_time_index.to_numpy(dtype=np.int64), np.arange(816)):
            raise ValueError(f"Incomplete or duplicate monthly coverage for seed {seed}")
        time = frame.global_time_index.to_numpy(dtype=np.int64)
        neighbors = nearest_neighbors(coordinates, time)
        for component, (true, pred) in _components(frame).items():
            target_std = float(np.std(true, ddof=0))
            if target_std <= 0:
                raise ValueError(f"Constant target component: {component}")
            model_rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
            model_rows.append({"space": "latent_prehead", "seed": seed,
                               "component": component, "target_std": target_std,
                               "model_rmse": model_rmse, "model_ratio": model_rmse / target_std})
            # Floor RMSE: predict the mean of the k nearest-neighbour truths.
            for k in K_VALUES:
                floor_rmse = float(np.sqrt(np.mean((true[neighbors[k]].mean(axis=1) - true) ** 2)))
                floor_rows.append({"space": "latent_prehead", "seed": seed,
                                   "component": component, "k": k, "target_std": target_std,
                                   "floor_rmse": floor_rmse, "floor_ratio": floor_rmse / target_std})
    return pd.DataFrame(floor_rows), pd.DataFrame(model_rows)


def summarize(floor: pd.DataFrame, model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-seed floor and model metrics; merge the K=10 floor with the model summary."""
    floor_summary = (floor.groupby(["space", "component", "k"], as_index=False)
                     .agg(seed_count=("seed", "nunique"),
                          floor_ratio_mean=("floor_ratio", "mean"),
                          floor_ratio_std=("floor_ratio", "std"),
                          floor_ratio_min=("floor_ratio", "min"),
                          floor_ratio_max=("floor_ratio", "max"),
                          floor_rmse_mean=("floor_rmse", "mean"),
                          floor_rmse_std=("floor_rmse", "std"),
                          target_std_mean=("target_std", "mean"))
                     .sort_values(["space", "component", "k"]).reset_index(drop=True))
    model_summary = (model.groupby(["space", "component"], as_index=False)
                     .agg(seed_count=("seed", "nunique"),
                          model_ratio_mean=("model_ratio", "mean"),
                          model_ratio_std=("model_ratio", "std"),
                          model_ratio_min=("model_ratio", "min"),
                          model_ratio_max=("model_ratio", "max"),
                          model_rmse_mean=("model_rmse", "mean"),
                          model_rmse_std=("model_rmse", "std"),
                          target_std_mean=("target_std", "mean"))
                     .sort_values(["space", "component"]).reset_index(drop=True))
    # Side-by-side comparison for K = 10 (the central neighbour count).
    comparison = (floor_summary.loc[floor_summary.k.eq(10)]
                  .merge(model_summary, on=["space", "component", "seed_count", "target_std_mean"],
                         validate="one_to_one")
                  .sort_values(["space", "component"]).reset_index(drop=True))
    return floor_summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-root", type=Path,
                        help="Root of the new trained M_20/fold_XX/seed_XXXXX runs")
    source.add_argument("--embeddings-npz", type=Path,
                        help="Previously exported pre-head embeddings")
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.campaign_root:
        if args.metadata_csv:
            parser.error("--metadata-csv is only used with --embeddings-npz")
        embeddings, metadata = export_embeddings(args.campaign_root)
    else:
        if not args.metadata_csv:
            parser.error("--embeddings-npz requires --metadata-csv")
        embeddings, metadata = load_embeddings(args.embeddings_npz, args.metadata_csv)
    floor, model = analyze(embeddings, metadata)
    summary, comparison = summarize(floor, model)
    args.output_dir.mkdir(parents=True)
    if args.campaign_root:
        np.savez_compressed(args.output_dir / "prehead_embeddings.npz", embeddings=embeddings)
        metadata.to_csv(args.output_dir / "prehead_metadata.csv", index=False)
    floor.to_csv(args.output_dir / "floor_per_seed.csv", index=False)
    model.to_csv(args.output_dir / "model_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "floor_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "floor_vs_model.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
