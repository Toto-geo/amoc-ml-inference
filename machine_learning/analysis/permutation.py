"""Source and longitude-bin permutation tests on held-out checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from machine_learning.dataset import SCALES, build_datasets
from machine_learning.model import build_model, load_checkpoint
from machine_learning.train import SEEDS


SURFACES = ("sst", "sss", "ssh", "obp")
# Every source bundles the tensors permuted together: Argo keeps the (profile, mask, lon, lat, time) group.
SOURCES = {"argo": ("x", "mask_padded", "lon", "lat", "days"),
           **{name: (name,) for name in SURFACES}}


def permutation_seed(*parts: object) -> int:
    """Derive a deterministic 31-bit permutation seed from any set of identifying parts."""
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def permutation_indices(n: int, seed: int) -> torch.Tensor:
    """Return a non-identity permutation of [0..n)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(n, generator=generator)
    if torch.equal(indices, torch.arange(n)):
        indices = indices.roll(1)                                  # avoid the rare identity permutation
    return indices


def permute(batch: dict[str, torch.Tensor], indices: torch.Tensor, *,
            source: str | None = None, bin_index: int | None = None) -> dict[str, torch.Tensor]:
    """Return a copy of batch with one source, or one longitude bin across surfaces, permuted in time."""
    if (source is None) == (bin_index is None):
        raise ValueError("Select exactly one source or longitude bin")
    changed = dict(batch)
    if source is not None:
        # Permute every tensor that belongs to the same source so they stay aligned.
        for key in SOURCES[source]:
            changed[key] = batch[key].index_select(0, indices)
    else:
        # Permute a single longitude column (same column index) across all four surface fields.
        width = batch["sst"].shape[1]
        if bin_index is None or not 1 <= bin_index <= width:
            raise ValueError("Longitude bin is outside the surface grid")
        column = bin_index - 1
        for key in SURFACES:
            value = batch[key].clone()
            value[:, column] = batch[key].index_select(0, indices)[:, column]
            changed[key] = value
    return changed


@torch.inference_mode()
def physical_mae(model, batch: dict[str, torch.Tensor], stats: dict) -> float:
    """Return the physical-space mean absolute error of the model on one batch."""
    model.eval()
    prediction, _ = model(batch)
    physical_prediction = prediction * stats["y_std_train"] + stats["y_mean_train"]
    physical_target = batch["y"] * stats["y_std_train"] + stats["y_mean_train"]
    result = float((physical_prediction - physical_target).abs().mean())
    if not np.isfinite(result):
        raise ValueError("Non-finite permutation metric")
    return result


def run(checkpoint_path: Path, *, regime: str, bins: int, fold: int, seed: int,
        sources: tuple[str, ...] = tuple(SOURCES), bin_indices: tuple[int, ...] = (),
        repeats: int = 10, source_base_seed: int = 90210,
        bin_base_seed: int = 314159,
        experiment_id: str | None = None) -> pd.DataFrame:
    """Evaluate one checkpoint, permute each requested source or bin, and return per-repeat metrics."""
    if (regime not in ("M", "Q") or bins not in SCALES or fold not in range(1, 9)
            or seed not in SEEDS or repeats < 1 or not sources and not bin_indices):
        raise ValueError("Invalid paper evaluation configuration")
    if set(sources) - set(SOURCES) or any(index not in range(1, bins + 1) for index in bin_indices):
        raise ValueError("Unknown source or longitude bin")
    experiment_id = experiment_id or ("exp1_M_full" if regime == "M" else "exp4_Q_quarter_mean_full")
    torch.set_num_threads(1)
    train, _, test, stats = build_datasets(regime, fold, n_bins=bins)
    model = build_model(train, initialize_assignment=False)
    checkpoint = load_checkpoint(model, checkpoint_path)
    if int(checkpoint["config"]["seed"]) != seed:
        raise ValueError("Checkpoint seed differs from the requested seed")
    # One large batch holding every test sample; this makes the baseline deterministic.
    batch = next(iter(DataLoader(test, batch_size=len(test), shuffle=False)))
    baseline = physical_mae(model, batch, stats)
    if baseline <= 0:
        raise ValueError("Baseline MAE must be positive")
    rows = []
    # Source-level permutations: shuffle an entire data source in time.
    for source in sources:
        for repeat in range(repeats):
            pseed = permutation_seed(source_base_seed, experiment_id, fold, seed, source, repeat)
            indices = permutation_indices(len(test), pseed)
            altered = physical_mae(model, permute(batch, indices, source=source), stats)
            rows.append(("source", source, 0, repeat, pseed, altered))
    # Bin-level permutations: shuffle a single longitude column across all four surface fields.
    for bin_index in bin_indices:
        for repeat in range(repeats):
            pseed = permutation_seed(bin_base_seed, experiment_id, fold, seed,
                                     "surface_bin", bin_index, repeat)
            indices = permutation_indices(len(test), pseed)
            altered = physical_mae(model, permute(batch, indices, bin_index=bin_index), stats)
            rows.append(("bin", "surface_block", bin_index, repeat, pseed, altered))
    return pd.DataFrame([
        {"regime": regime, "bins": bins, "fold": fold, "seed": seed,
         "kind": kind, "source": source, "bin_index": index, "repeat": repeat,
         "permutation_seed": pseed, "baseline_mae_phys": baseline,
         "permuted_mae_phys": altered,
         "mae_increase_pct": 100 * (altered - baseline) / baseline}
        for kind, source, index, repeat, pseed, altered in rows
    ])


def summarize(frame: pd.DataFrame, *, expected_folds: int = 8,
              expected_seeds: int = 20, expected_repeats: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average the per-repeat MAE increase across repeats, folds and seeds."""
    keys = ["regime", "bins", "kind", "source", "bin_index"]
    runs = keys + ["fold", "seed"]
    if frame.empty or frame.duplicated(runs + ["repeat"]).any():
        raise ValueError("Empty or duplicate permutation records")
    # Check that every (kind, source, bin_index) group has every expected fold, seed and repeat.
    for _, group in frame.groupby(keys):
        if sorted(group.fold.unique()) != list(range(1, expected_folds + 1)):
            raise ValueError("Incomplete permutation folds")
        for _, fold in group.groupby("fold"):
            if fold.seed.nunique() != expected_seeds:
                raise ValueError("Incomplete permutation seeds")
            for _, run_rows in fold.groupby("seed"):
                if sorted(run_rows.repeat) != list(range(expected_repeats)):
                    raise ValueError("Incomplete permutation repeats")
    fold_mean = frame.groupby(keys + ["fold"], as_index=False).mae_increase_pct.mean()
    summary = (fold_mean.groupby(keys, as_index=False)
               .agg(mean_pct=("mae_increase_pct", "mean"),
                    std_pct=("mae_increase_pct", "std")))
    return fold_mean, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assess = commands.add_parser("run", help="Evaluate one trusted checkpoint")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--output-dir", type=Path, required=True)
    assess.add_argument("--regime", choices=("M", "Q"), required=True)
    assess.add_argument("--bins", type=int, choices=tuple(SCALES), required=True)
    assess.add_argument("--fold", type=int, required=True)
    assess.add_argument("--seed", type=int, required=True)
    assess.add_argument("--sources", nargs="*", choices=tuple(SOURCES), default=list(SOURCES))
    assess.add_argument("--bin-indices", nargs="*", type=int, default=[])
    assess.add_argument("--repeats", type=int, default=10)
    assess.add_argument("--experiment-id", help="Historical campaign ID for the permutation seed schedule")
    aggregate = commands.add_parser("aggregate", help="Summarize a complete campaign")
    aggregate.add_argument("permutations", type=Path, nargs="+")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--expected-folds", type=int, default=8)
    aggregate.add_argument("--expected-seeds", type=int, default=20)
    aggregate.add_argument("--expected-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.checkpoint, regime=args.regime, bins=args.bins, fold=args.fold,
                     seed=args.seed, sources=tuple(args.sources), bin_indices=tuple(args.bin_indices),
                     repeats=args.repeats, experiment_id=args.experiment_id)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result.to_csv(args.output_dir / "permutation_metrics.csv", index=False)
        print(result.to_string(index=False))
    else:
        frame = pd.concat([pd.read_csv(path) for path in args.permutations], ignore_index=True)
        fold_mean, summary = summarize(frame, expected_folds=args.expected_folds,
                                       expected_seeds=args.expected_seeds,
                                       expected_repeats=args.expected_repeats)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        fold_mean.to_csv(args.output_dir / "fold_permutation.csv", index=False)
        summary.to_csv(args.output_dir / "permutation_summary.csv", index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
