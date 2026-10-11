"""Figure 6: eight-panel reconstruction and skill summary.

Top two panels: quarterly and monthly reconstructed series overlaid on the
ground truth, with fold boundaries marked. Bottom six panels: fold-level
mean/std bars for RMSE, MAE, bias, Pearson r, R2 and spread ratio, comparing
the base model against Ridge and seasonal-climatology baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from machine_learning.dataset import fold_indices
from machine_learning.evaluate import aggregate_predictions, skill


COLORS = {"truth": "#111111", "base": "#b22222", "ridge": "#2f6b3f",
          "climatology": "#7f7f7f", "fold_boundary": "#111111",
          "fold_label": "#8a9099", "error": "0.25"}
METRICS = ("rmse", "mae", "pearson", "r2", "spread_ratio", "bias")


def prepare(latent: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse latent seeds to a seed mean and compute per-fold skill for every series."""
    # Accept either raw seed-level predictions (will be averaged) or pre-aggregated seed means.
    if "experiment_id" in latent.columns:
        mapping = {"exp1_M_full": "M", "exp4_Q_quarter_mean_full": "Q"}
        latent = latent.loc[latent.experiment_id.isin(mapping)].copy()
        latent["regime"] = latent.experiment_id.map(mapping)
        latent["bins"] = 20
    if "seed" in latent.columns:
        mean, _, _ = aggregate_predictions(latent)
    else:
        required_mean = {"regime", "bins", "fold", "global_time_index", "date",
                         "y_true_phys", "y_pred_phys"}
        if required_mean - set(latent) or latent[list(required_mean)].isna().any().any():
            raise ValueError("Incomplete seed-mean predictions")
        mean = latent.copy()
        if set(zip(mean.regime, mean.bins)) != {("M", 20), ("Q", 20)}:
            raise ValueError("Expected final M and Q at 20 bins")
        # Check that every (regime, fold) block holds the right test indices.
        for regime, count in (("M", 816), ("Q", 272)):
            block = mean.loc[mean.regime.eq(regime)]
            if len(block) != count or not np.array_equal(
                    np.sort(block.global_time_index.to_numpy()), np.arange(count)):
                raise ValueError(f"Incomplete {regime} seed-mean coverage")
            for fold in range(1, 9):
                if not np.array_equal(
                        np.sort(block.loc[block.fold.eq(fold), "global_time_index"].to_numpy()),
                        fold_indices(count, fold)["test"]):
                    raise ValueError(f"Incorrect {regime} fold {fold} coverage")
        mean = mean.sort_values(["regime", "global_time_index"]).reset_index(drop=True)
    mean["date"] = pd.to_datetime(mean.date)
    baseline = baseline.copy()
    baseline["date"] = pd.to_datetime(baseline.date)
    required = {"series_id", "regime", "fold", "global_time_index", "date",
                "y_true_phys", "y_pred_phys"}
    if required - set(baseline) or baseline.series_id.isna().any():
        raise ValueError("Incomplete baseline predictions")
    # Per-fold skill rows for the latent model (M and Q).
    rows = []
    for regime in ("M", "Q"):
        part = mean.loc[mean.regime.eq(regime) & mean.bins.eq(20)]
        series_id = "latent_M_full" if regime == "M" else "latent_Q_quarter_mean"
        for fold, block in part.groupby("fold"):
            rows.append({"series_id": series_id, "fold": int(fold),
                         **skill(block.y_true_phys.to_numpy(), block.y_pred_phys.to_numpy())})
    expected = {"ridge_M", "ridge_Q", "climatology_M", "climatology_Q"}
    if set(baseline.series_id) != expected:
        raise ValueError("Expected the four final baseline series")
    if baseline.duplicated(["series_id", "global_time_index"]).any():
        raise ValueError("Duplicate baseline predictions")
    # Per-fold skill rows for the four baseline series; truth must match the latent reference.
    for series_id, series in baseline.groupby("series_id"):
        regime = str(series.regime.iloc[0])
        reference = mean.loc[mean.regime.eq(regime) & mean.bins.eq(20)]
        aligned = series.sort_values("global_time_index").reset_index(drop=True)
        reference = reference.sort_values("global_time_index").reset_index(drop=True)
        if (not np.array_equal(aligned.global_time_index, reference.global_time_index)
                or not np.array_equal(aligned.fold, reference.fold)
                or not np.array_equal(aligned.date.dt.normalize(), reference.date.dt.normalize())
                or not np.allclose(aligned.y_true_phys, reference.y_true_phys,
                                   rtol=0, atol=2e-6)):
            raise ValueError(f"Baseline target or time coverage differs: {series_id}")
        for fold, block in series.groupby("fold"):
            rows.append({"series_id": str(series_id), "fold": int(fold),
                         **skill(block.y_true_phys.to_numpy(), block.y_pred_phys.to_numpy())})
    metrics = pd.DataFrame(rows)
    if len(metrics) != 48 or any(set(part.fold) != set(range(1, 9))
                                 for _, part in metrics.groupby("series_id")):
        raise ValueError("Expected eight folds for each of six series")
    return mean, metrics


def plot_metric_bars(ax, metrics: pd.DataFrame, metric: str, title: str) -> None:
    """Grouped bar chart for one metric: six series (3 M + 3 Q) with across-fold std error bars."""
    # Six (series, colour, x-position) tuples: three for 30-day (M) and three for 90-day (Q).
    specs = (("latent_M_full", COLORS["base"]), ("ridge_M", COLORS["ridge"]),
             ("climatology_M", COLORS["climatology"]),
             ("latent_Q_quarter_mean", COLORS["base"]), ("ridge_Q", COLORS["ridge"]),
             ("climatology_Q", COLORS["climatology"]))
    for (series_id, color), x in zip(specs, (-0.16, 0.0, 0.16, 0.84, 1.0, 1.16)):
        vals = metrics.loc[metrics.series_id.eq(series_id), metric].astype(float)
        ax.bar(x, float(vals.mean()), width=0.12, color=color, edgecolor="0.25",
               linewidth=0.5, yerr=float(vals.std(ddof=1)), capsize=2.5,
               error_kw={"elinewidth": 0.8, "ecolor": COLORS["error"]})
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["30 days", "90 days"])
    ax.set_title(title, loc="left", pad=6, fontsize=13.4)
    ax.grid(False)
    ax.set_xlim(-0.36, 1.36)


def _pad_ylim(ax) -> None:
    """Add 8% of the current y-range as padding above and below."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymin), 1.0)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)


def _axis_tight_center_x(fig, ax) -> float:
    """Return the horizontal centre of an axis's tight bounding box in figure coordinates."""
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    return float(bbox.x0 + 0.5 * bbox.width)


def build_figure(mean: pd.DataFrame, metrics: pd.DataFrame):
    """Compose the eight-panel figure and return it."""
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    # Truth and prediction, sorted by date, for both regimes.
    truth_m = mean.loc[mean.regime.eq("M"), ["date", "fold", "y_true_phys"]].sort_values("date")
    truth_q = mean.loc[mean.regime.eq("Q"), ["date", "fold", "y_true_phys"]].sort_values("date")
    # Fold boundaries and the mid-fold year ticks shown on the monthly panel.
    fold_ranges_m = truth_m.groupby("fold")["date"].agg(["min", "max"]).reset_index()
    fold_boundary_dates = fold_ranges_m.loc[fold_ranges_m.fold.gt(1), "min"].tolist()
    fold_boundary_dates.append(fold_ranges_m.iloc[-1]["max"])
    fold_year_tick_dates = []
    for _, row in fold_ranges_m.iterrows():
        fold_year_tick_dates.extend([row["min"], row["min"] + (row["max"] - row["min"]) / 2])
    fold_year_tick_dates.append(fold_ranges_m.iloc[-1]["max"])
    fold_year_tick_dates = sorted(pd.to_datetime(pd.Series(fold_year_tick_dates)).drop_duplicates())
    fold_year_tick_labels = [str(pd.Timestamp(date).year) for date in fold_year_tick_dates]

    # Grid: two stacked time-series rows on top, six bar panels (2 x 3) at the bottom.
    fig = plt.figure(figsize=(15.5, 10.35))
    outer_gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.65, 1.0, 1.0], hspace=0.46)
    time_gs = outer_gs[0].subgridspec(nrows=2, ncols=1, height_ratios=[1.05, 1.45], hspace=0.16)
    bar_gs = outer_gs[1:].subgridspec(nrows=2, ncols=6, hspace=0.46, wspace=0.42)
    ax_q = fig.add_subplot(time_gs[0, 0])
    ax_m = fig.add_subplot(time_gs[1, 0], sharex=ax_q)
    bar_axes = [fig.add_subplot(bar_gs[0, 0:2]), fig.add_subplot(bar_gs[0, 2:4]),
                fig.add_subplot(bar_gs[0, 4:6]), fig.add_subplot(bar_gs[1, 0:2]),
                fig.add_subplot(bar_gs[1, 2:4]), fig.add_subplot(bar_gs[1, 4:6])]
    for ax in (ax_q, ax_m):
        ax.set_zorder(2)
        ax.set_facecolor("none")

    # Panel (a): quarterly truth and prediction, with per-fold centred labels at the top.
    ax_q.plot(truth_q.date, truth_q.y_true_phys, color=COLORS["truth"], linewidth=1.22,
              label="Ground truth", zorder=5)
    ax_q.plot(truth_q.date, mean.loc[mean.regime.eq("Q")].sort_values("date").y_pred_phys,
              color=COLORS["base"], linewidth=1.22, label="Base model", zorder=5)
    ax_q.set_ylabel("Transport [Sv]", fontsize=11.8)
    ax_q.set_title("a) Quarterly reconstruction", loc="left", pad=8, fontsize=13.4)
    ax_q.grid(axis="y", color="0.9", linewidth=0.6)
    _pad_ylim(ax_q)
    ax_q.tick_params(axis="y", labelsize=10.5)
    ax_q.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
    ax_q.spines["bottom"].set_visible(False)
    for _, row in fold_ranges_m.iterrows():
        center = row["min"] + (row["max"] - row["min"]) / 2
        ax_q.text(center, 1.0, f"Test block {int(row['fold'])}",
                  transform=ax_q.get_xaxis_transform(), ha="center", va="top",
                  color=COLORS["fold_label"], fontsize=9.4, alpha=0.78, zorder=6)

    # Panel (b): monthly truth and prediction, with mid-fold year ticks on the x axis.
    ax_m.plot(truth_m.date, truth_m.y_true_phys, color=COLORS["truth"], linewidth=1.22,
              label="Ground truth", zorder=5)
    ax_m.plot(truth_m.date, mean.loc[mean.regime.eq("M")].sort_values("date").y_pred_phys,
              color=COLORS["base"], linewidth=1.22, label="Base model", zorder=5)
    ax_m.set_ylabel("Transport [Sv]", fontsize=11.8)
    ax_m.set_title("b) Monthly reconstruction", loc="left", pad=7, fontsize=13.4)
    ax_m.grid(axis="y", color="0.9", linewidth=0.6)
    _pad_ylim(ax_m)
    ax_m.set_xticks(fold_year_tick_dates)
    ax_m.set_xticklabels(fold_year_tick_labels)
    ax_m.tick_params(axis="both", labelsize=10.5)
    ax_m.set_xlabel("Years", labelpad=8, fontsize=11.8)
    ax_m.set_xlim(pd.Timestamp("1958-01-01"), pd.Timestamp("2025-12-31"))
    # Panels (c-h): six bar charts, one per metric.
    panels = (("rmse", "RMSE [Sv] (lower better)"), ("mae", "MAE [Sv] (lower better)"),
              ("bias", "Bias [Sv] (closer to 0)"),
              ("pearson", r"Pearson $r$ (higher better)"),
              ("r2", r"$R^2$ (higher better)"),
              ("spread_ratio", "Spread ratio (closer to 1)"))
    for ax, (metric, title), letter in zip(bar_axes, panels, "cdefgh"):
        plot_metric_bars(ax, metrics, metric, f"{letter}) {title}")
        ax.tick_params(axis="both", labelsize=10.5)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.965, bottom=0.11)
    # Dashed fold-boundary verticals that span both time-series panels, with a gap behind the (b) title.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_box, bottom_box = ax_q.get_position(), ax_m.get_position()
    boundary_box = [top_box.x0, bottom_box.y0, top_box.width, top_box.y1 - bottom_box.y0]
    title_box = ax_m.title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    title_y_lower = (title_box.y0 - boundary_box[1]) / boundary_box[3] - 0.020
    title_y_upper = (title_box.y1 - boundary_box[1]) / boundary_box[3] + 0.055
    boundary_ax = fig.add_axes(boundary_box, zorder=0, frameon=False)
    boundary_ax.set_xlim(ax_m.get_xlim())
    boundary_ax.set_ylim(0, 1)
    boundary_ax.axis("off")
    line_kwargs = dict(color=COLORS["fold_boundary"], linestyle="--",
                       linewidth=0.7, alpha=0.55, zorder=0)
    for boundary in fold_boundary_dates:
        if boundary == fold_boundary_dates[0]:
            # Split the first boundary around the title text to avoid crossing it.
            boundary_ax.plot([boundary, boundary], [0.0, max(0.0, title_y_lower)], **line_kwargs)
            boundary_ax.plot([boundary, boundary], [min(1.0, title_y_upper), 1.0], **line_kwargs)
        else:
            boundary_ax.axvline(boundary, **line_kwargs)

    # Legend: place a reference legend at a small font to measure its height, then
    # draw the final one at the publication size with the anchor adjusted.
    legend_center_x = _axis_tight_center_x(fig, bar_axes[4])
    legend_handles = [Line2D([0], [0], color=COLORS["truth"], lw=1.5, label="Ground truth"),
                      Line2D([0], [0], color=COLORS["base"], lw=1.5, label="Base model"),
                      Patch(facecolor=COLORS["ridge"], edgecolor="0.25", label="Ridge"),
                      Patch(facecolor=COLORS["climatology"], edgecolor="0.25", label="Climatology")]
    legend_anchor_y = 0.012
    legend_kwargs = dict(handles=legend_handles, loc="lower center", ncol=2,
                         handlelength=1.9, handletextpad=0.55, columnspacing=1.25,
                         borderpad=0.5, labelspacing=0.45, frameon=True, framealpha=0.86,
                         facecolor="white", edgecolor="0.75", borderaxespad=0.0)
    reference_legend = fig.legend(**legend_kwargs, prop={"size": 10.9},
                                  bbox_to_anchor=(legend_center_x, legend_anchor_y))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reference_top = reference_legend.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
    reference_legend.remove()
    legend = fig.legend(**legend_kwargs, prop={"size": 13.4},
                        bbox_to_anchor=(legend_center_x, legend_anchor_y))
    fig.canvas.draw()
    legend_top = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
    adjusted_anchor_y = legend_anchor_y + reference_top - legend_top
    if abs(adjusted_anchor_y - legend_anchor_y) > 1e-4:
        legend.remove()
        fig.legend(**legend_kwargs, prop={"size": 13.4},
                   bbox_to_anchor=(legend_center_x, adjusted_anchor_y))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Final M/Q seed-mean evaluation CSV or seed-level historical CSV")
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mean, metrics = prepare(pd.read_csv(args.predictions), pd.read_csv(args.baselines))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    fig = build_figure(mean, metrics)
    fig.savefig(args.output_dir / "fig06_reconstruction.pdf",
                bbox_inches="tight")
    metrics.to_csv(args.output_dir / "reconstruction_fold_metrics.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
