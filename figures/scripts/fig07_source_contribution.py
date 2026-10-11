"""Figure 7: source-permutation sensitivity plot.

Lollipop plot summarising the mean MAE increase (in percent) when each data
source is permuted in time, for both the monthly (30 days) and quarterly
(90 days) models. One row per source; the two time scales are drawn as
small vertical offsets around the row centre.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


SOURCES = ("argo", "sss", "ssh", "obp", "sst")
LABELS = {key: key.upper() if key != "argo" else "Argo" for key in SOURCES}
TIMES = (("M", "30 days", "o"), ("Q", "90 days", "s"))      # (regime, label, marker)


def plot_data(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the 20-bin source-permutation summaries for M and Q and attach plot labels."""
    required = {"regime", "bins", "kind", "source", "bin_index", "mean_pct"}
    if required - set(summary):
        raise ValueError(f"Missing permutation-summary columns: {sorted(required - set(summary))}")
    selected = summary.loc[summary.kind.eq("source") & summary.bins.eq(20)].copy()
    if (len(selected) != 10 or selected.duplicated(["regime", "source"]).any()
            or set(selected.regime) != {"M", "Q"}
            or any(set(group.source) != set(SOURCES) for _, group in selected.groupby("regime"))
            or not np.isfinite(selected.mean_pct).all()):
        raise ValueError("Expected five complete source summaries for both final regimes")
    selected["time_scale"] = selected.regime.map({"M": "30 days", "Q": "90 days"})
    selected["source_label"] = selected.source.map(LABELS)
    return selected


def build_figure(data: pd.DataFrame):
    """Compose the lollipop figure: horizontal rows per source, two markers per row."""
    mpl.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5,
                         "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 9.2,
                         "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    # Sort sources by mean sensitivity across the two time scales (most impactful on top).
    source_order = (data.groupby("source").mean_pct.mean()
                    .sort_values(ascending=False).index.tolist())
    y_positions = {key: idx for idx, key in enumerate(source_order)}
    y_offsets = {"30 days": -0.13, "90 days": 0.13}       # small vertical split between the two time scales
    x_min = min(-1.0, float(data.mean_pct.min()) - 1.0)
    x_max = float(data.mean_pct.max()) + 3.0
    fig, ax = plt.subplots(figsize=(8.35, 4.5))
    # Connecting line between the two markers of each source.
    for source in source_order:
        pair = data.loc[data.source.eq(source)].set_index("time_scale").loc[["30 days", "90 days"]]
        ax.plot(pair.mean_pct, [y_positions[source] + y_offsets[key] for key in pair.index],
                color="#111111", linewidth=1.05, alpha=0.52, zorder=2)
    # Marker per (source, time scale) with a different glyph per regime.
    for regime, time_scale, marker in TIMES:
        for _, row in data.loc[data.regime.eq(regime)].iterrows():
            ax.scatter(row.mean_pct, y_positions[row.source] + y_offsets[time_scale],
                       s=42, marker=marker, color="#111111", edgecolor="white",
                       linewidth=0.65, zorder=3)
    ax.axvline(0.0, color="0.25", linewidth=0.9, linestyle="-", zorder=1)
    ax.grid(axis="x", color="0.88", linewidth=0.7, linestyle="-", zorder=0)
    ax.set_yticks(range(len(source_order)))
    ax.set_yticklabels([LABELS[key] for key in source_order])
    ax.set_ylim(len(source_order) - 0.55, -0.55)       # invert y-axis so most impactful source sits on top
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _pos: "0" if abs(value) < 1e-10 else f"{value:g}"))
    ax.set_xlabel("Mean MAE increase after source permutation [%]", labelpad=7)
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [Line2D([0], [0], marker=marker, color="0.20", linestyle="None",
                      markersize=6.0, label=time_scale) for _, time_scale, marker in TIMES]
    fig.subplots_adjust(left=0.16, right=0.985, top=0.95, bottom=0.25)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
               framealpha=0.94, facecolor="white", edgecolor="0.76",
               handletextpad=0.45, columnspacing=1.35, borderpad=0.50,
               bbox_to_anchor=(0.5725, 0.035))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = plot_data(pd.read_csv(args.permutation_summary))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    fig = build_figure(data)
    fig.savefig(args.output_dir / "fig07_source_contribution.pdf", bbox_inches="tight")
    data.to_csv(args.output_dir / "source_plot_data.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
