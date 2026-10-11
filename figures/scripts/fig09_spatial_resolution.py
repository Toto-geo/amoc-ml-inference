"""Figure 9: spatial-resolution skill curves from per-run M/Q checkpoint metrics.

Three panels of fold-mean metrics (RMSE, Pearson r, spread ratio) versus the
number of longitude bins (20, 40, 80, 160) for both regimes. Error bars show
the across-fold standard deviation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from machine_learning.train import SEEDS


BINS = (20, 40, 80, 160)
REGIMES = ("M", "Q")
METRICS = (("test_rmse_phys", "a) RMSE [Sv]"),
           ("test_pearson", r"b) Pearson $r$"),
           ("test_spread_ratio", "c) Spread ratio"))
STYLES = {
    "M": {"label": "30 days", "color": "#111111", "marker": "o",
          "x_offset": -0.045, "error_cap_color": "0.16", "error_bar_color": "0.42"},
    "Q": {"label": "90 days", "color": "#555555", "marker": "s",
          "x_offset": 0.045, "error_cap_color": "0.46", "error_bar_color": "0.68"},
}


def load_runs(runs_root: Path) -> pd.DataFrame:
    """Walk the runs tree and collect per-run test metrics from summary.json."""
    rows = []
    for regime in REGIMES:
        for bins in BINS:
            for fold in range(1, 9):
                for seed in SEEDS:
                    path = (runs_root / f"{regime}_{bins}" / f"fold_{fold:02d}"
                            / f"seed_{seed:05d}" / "summary.json")
                    with path.open() as handle:
                        saved = json.load(handle)
                    # Accept both the historical layout and the new trainer's layout.
                    historical = "config" not in saved
                    config = ({"regime": saved["regime"],
                               "bins": saved["dataset_model_contract"]["surface_n_bins"],
                               "fold": saved["fold"], "seed": saved["seed"]}
                              if historical else saved["config"])
                    if (config["regime"], int(config["bins"]), int(config["fold"]),
                            int(config["seed"])) != (regime, bins, fold, seed):
                        raise ValueError(f"Run metadata differs from path: {path}")
                    if saved.get("test_used_for_model_selection") is not False:
                        raise ValueError(f"Test data used for checkpoint selection: {path}")
                    metrics = saved["test_metrics"]
                    prefix = "test_" if historical else ""
                    rows.append({"regime": regime, "surface_n_bins": bins,
                                 "fold": fold, "seed": seed,
                                 "test_rmse_phys": metrics[f"{prefix}rmse_phys"],
                                 "test_pearson": metrics[f"{prefix}pearson"],
                                 "test_spread_ratio": metrics[f"{prefix}spread_ratio"]})
    return pd.DataFrame(rows)


def plot_data(runs: pd.DataFrame) -> pd.DataFrame:
    """Average per-seed metrics within each fold, then return fold-mean and fold-std rows."""
    required = {"regime", "surface_n_bins", "fold", "seed", *(name for name, _ in METRICS)}
    if required - set(runs) or runs[list(required)].isna().any().any():
        raise ValueError("Incomplete per-run spatial-resolution metrics")
    runs = runs.loc[:, list(required)].copy()
    runs["surface_n_bins"] = runs.surface_n_bins.astype(int)
    # Expect 2 regimes x 4 resolutions x 8 folds x 20 seeds = 1 280 runs.
    if len(runs) != 2 * 4 * 8 * len(SEEDS) or runs.duplicated(
            ["regime", "surface_n_bins", "fold", "seed"]).any():
        raise ValueError("Expected exactly 1,280 distinct paper runs")
    if not np.isfinite(runs[[name for name, _ in METRICS]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite test metric")
    for (regime, bins), group in runs.groupby(["regime", "surface_n_bins"]):
        if regime not in REGIMES or bins not in BINS or set(group.fold) != set(range(1, 9)):
            raise ValueError("Unexpected regime, resolution, or fold")
        if any(set(part.seed) != set(SEEDS) for _, part in group.groupby("fold")):
            raise ValueError(f"Incomplete seed coverage: {regime}/{bins}")
    folded = runs.groupby(["regime", "surface_n_bins", "fold"], as_index=False)[
        [name for name, _ in METRICS]].mean()
    # Fold-level mean and std per (regime, resolution, metric).
    rows = []
    for (regime, bins), group in folded.groupby(["regime", "surface_n_bins"]):
        for metric, _ in METRICS:
            values = group[metric].to_numpy(dtype=float)
            rows.append({"regime": regime, "surface_n_bins": bins,
                         "metric": metric, "mean": float(values.mean()),
                         "fold_std": float(values.std(ddof=1))})
    return pd.DataFrame(rows).sort_values(
        ["regime", "surface_n_bins", "metric"]).reset_index(drop=True)


def build_figure(data: pd.DataFrame):
    """Compose three error-bar panels, one per metric, with M and Q offset slightly on x."""
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 240,
                         "font.size": 9.0, "axes.titlesize": 9.6,
                         "axes.labelsize": 9.4, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 8.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "normal"})
    fig, axes = plt.subplots(1, 3, figsize=(8.42, 3.15), sharex=True)
    x = np.arange(len(BINS))                                  # one tick per resolution
    for ax, (metric, title) in zip(axes, METRICS):
        for regime in REGIMES:
            part = data.loc[data.regime.eq(regime) & data.metric.eq(metric)].sort_values(
                "surface_n_bins")
            if tuple(part.surface_n_bins) != BINS:
                raise ValueError(f"Missing plotted resolutions: {regime}/{metric}")
            style = STYLES[regime]
            container = ax.errorbar(
                x + style["x_offset"], part["mean"].to_numpy(dtype=float),
                yerr=part.fold_std.to_numpy(dtype=float),
                color=style["color"], ecolor=style["error_cap_color"],
                marker=style["marker"], linewidth=1.25, markersize=3.7,
                markeredgewidth=0.8, elinewidth=0.75, capsize=2.2,
                capthick=0.85, label=style["label"])
            # Dim the error-bar segments slightly (keep caps at full tone).
            for barline in container.lines[2]:
                barline.set_color(style["error_bar_color"])
                barline.set_linewidth(0.75)
        if metric == "test_spread_ratio":
            ax.axhline(1.0, color="0.62", linestyle=":", linewidth=0.7, zorder=0)     # ideal spread = 1
        ax.set_title(title, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(value) for value in BINS])
        ax.set_xlabel("Number of bins")
        ax.grid(axis="y", color="0.93", linewidth=0.55)
    handles = [Line2D([0], [0], color=STYLES[regime]["color"],
                      marker=STYLES[regime]["marker"], linewidth=1.25,
                      markersize=4.4, label=STYLES[regime]["label"])
               for regime in REGIMES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
               framealpha=0.94, facecolor="white", edgecolor="0.76",
               handletextpad=0.45, columnspacing=1.35, borderpad=0.42,
               bbox_to_anchor=(0.5, 0.055), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.145, 1, 1.0))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-root", type=Path, help="New trainer output root")
    source.add_argument("--run-metrics", type=Path, help="Per-run metric CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.runs_root) if args.runs_root else pd.read_csv(args.run_metrics)
    data = plot_data(runs)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    fig = build_figure(data)
    fig.savefig(args.output_dir / "fig09_spatial_resolution.pdf",
                bbox_inches="tight")
    data.to_csv(args.output_dir / "resolution_plot_data.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
