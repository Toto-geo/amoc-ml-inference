#!/usr/bin/env python
# coding: utf-8
"""Figure 5: single-branch surface encoder schematic with gating.

Diagram of the per-variable surface pipeline: the gridded predictor input, two
hidden layers, the latent output, the seasonal gate and the gated latent. Two
bottom brackets separate the encoding and gating stages.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


parser = argparse.ArgumentParser(description="Recreate paper Figure 5: surface encoder")
parser.add_argument("--output-dir", type=Path, required=True)
OUTPUT_DIR = parser.parse_args().output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    }
)

COLORS = {
    "ink": "#111111",
    "accent": "#8f1d1d",       # input/latent node ring
    "muted": "#6e7681",
    "bg": "#ffffff",
    "gate_edge": "#2f5f99",    # seasonal-gate square
    "gate_fill": "#e7f0fb",
}

OUTPUT_BASENAME = OUTPUT_DIR / "fig05_surface_encoder"
print(OUTPUT_BASENAME)




def draw_node(ax, x, y, r=0.25, *, edgecolor="#111111", text="", lw=1.3, fontsize=12.0):
    """Draw one white circular node with an optional centred text label."""
    circ = Circle((x, y), radius=r, facecolor="white", edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(circ)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])
    return circ


def connect_layers(ax, xs, ys_left, ys_right, *, lw=0.8, alpha=0.8):
    """Draw a fully-connected lattice of thin lines between two vertical node columns."""
    x0, x1 = xs
    for y0 in ys_left:
        for y1 in ys_right:
            ax.plot([x0, x1], [y0, y1], color=COLORS["ink"], lw=lw, alpha=alpha, zorder=0)


def arrow(ax, start, end, *, lw=1.2):
    """Draw a thin arrow from start to end."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=COLORS["ink"], mutation_scale=12),
    )


def add_top_label(ax, x, text):
    """Place a two-line column title above x."""
    ax.text(x, 7.72, text, ha="center", va="center", fontsize=18, color=COLORS["ink"], linespacing=0.92)


def add_bottom_note(ax, x, text):
    """Place a small muted note below x."""
    ax.text(x, 3.15, text, ha="center", va="top", fontsize=13, color=COLORS["muted"], linespacing=1.10)


def draw_bottom_brackets(ax, x_start, x_mid, x_end, *, y=3.00, tick_height=0.48, lw=1.4, fontsize=18):
    """Draw two side-by-side U brackets labelled 'encoding' and 'gating'."""
    color = COLORS["muted"]
    ax.plot([x_start, x_end], [y, y], color=color, lw=lw, solid_capstyle="butt")
    for x, height in ((x_start, tick_height * 0.75), (x_mid, tick_height * 0.5), (x_end, tick_height * 0.75)):
        ax.plot([x, x], [y, y + height], color=color, lw=lw, solid_capstyle="butt")
    ax.text((x_start + x_mid) / 2, y - 0.18, "encoding", ha="center", va="top", fontsize=fontsize, color=color)
    ax.text((x_mid + x_end) / 2, y - 0.18, "gating", ha="center", va="top", fontsize=fontsize, color=color)


def draw_gate_square(ax, x, y, size, text):
    """Draw the seasonal-gate box with a centred label."""
    rect = Rectangle((x - size / 2, y - size / 2), size, size, facecolor=COLORS["gate_fill"], edgecolor=COLORS["gate_edge"], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=14.0, color=COLORS["ink"])
    return rect


def draw_vector_column(ax, x, ys, labels, *, color="#111111", lw=1.4, tick=0.13, pad=0.34, fontsize=13.0):
    """Draw a column of labels inside a vertical bracket pair (the gated-latent vector)."""
    for y, label in zip(ys, labels):
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=color)
    top = ys[0] + 0.28
    bottom = ys[-1] - 0.28
    x_left = x - pad
    x_right = x + pad
    # Left bracket.
    ax.plot([x_left, x_left], [bottom, top], color=color, lw=lw)
    ax.plot([x_left, x_left + tick], [top, top], color=color, lw=lw)
    ax.plot([x_left, x_left + tick], [bottom, bottom], color=color, lw=lw)
    # Right bracket.
    ax.plot([x_right, x_right], [bottom, top], color=color, lw=lw)
    ax.plot([x_right - tick, x_right], [top, top], color=color, lw=lw)
    ax.plot([x_right - tick, x_right], [bottom, bottom], color=color, lw=lw)
    return (x_left, x_right, bottom, top)


def draw_predictor_v_gated_latent_brackets():
    """Compose the full schematic: input, two hidden layers, latent output, gate, gated latent."""
    fig, ax = plt.subplots(figsize=(14.8, 7.15))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0.70, 14.15)
    ax.set_ylim(2.38, 7.95)
    ax.axis("off")

    # Column x positions and shared row y positions.
    x_in, x_h1, x_h2, x_lat, x_gate, x_final = 1.0, 4.0, 6.9, 9.5, 11.55, 13.8
    y_in = [6.95, 5.95, 3.95]
    y_h1 = [6.95, 5.95, 3.95]
    y_h2 = [6.95, 5.95, 3.95]
    y_vec = [6.95, 5.95, 4.95, 3.95]
    y_ellipsis = 4.95

    # Fully-connected lines between adjacent layers.
    connect_layers(ax, (x_in + 0.25, x_h1 - 0.25), y_in, y_h1)
    connect_layers(ax, (x_h1 + 0.25, x_h2 - 0.25), y_h1, y_h2)
    connect_layers(ax, (x_h2 + 0.25, x_lat - 0.25), y_h2, y_vec)

    # Input column with three visible nodes and an ellipsis.
    draw_node(ax, x_in, y_in[0], edgecolor=COLORS["accent"], text=r"$x_{v,1}$")
    draw_node(ax, x_in, y_in[1], edgecolor=COLORS["accent"], text=r"$x_{v,2}$")
    draw_node(ax, x_in, y_in[2], edgecolor=COLORS["accent"], text=r"$x_{v,20}$")
    ax.text(x_in, y_ellipsis, r"$\cdots$", ha="center", va="center", fontsize=28, color=COLORS["ink"])

    draw_node(ax, x_h1, y_h1[0], text=r"$h^{(1)}_1$")
    draw_node(ax, x_h1, y_h1[1], text=r"$h^{(1)}_2$")
    draw_node(ax, x_h1, y_h1[2], text=r"$h^{(1)}_{32}$")
    ax.text(x_h1, y_ellipsis, r"$\cdots$", ha="center", va="center", fontsize=28, color=COLORS["ink"])

    draw_node(ax, x_h2, y_h2[0], text=r"$h^{(2)}_1$")
    draw_node(ax, x_h2, y_h2[1], text=r"$h^{(2)}_2$")
    draw_node(ax, x_h2, y_h2[2], text=r"$h^{(2)}_{32}$")
    ax.text(x_h2, y_ellipsis, r"$\cdots$", ha="center", va="center", fontsize=28, color=COLORS["ink"])

    latent_labels = [r"$z_{v,1}$", r"$z_{v,2}$", r"$z_{v,3}$", r"$z_{v,4}$"]
    final_labels = [r"$\tilde{z}_{v,1}$", r"$\tilde{z}_{v,2}$", r"$\tilde{z}_{v,3}$", r"$\tilde{z}_{v,4}$"]

    for yy, label in zip(y_vec, latent_labels):
        draw_node(ax, x_lat, yy, edgecolor=COLORS["accent"], text=label)

    draw_gate_square(ax, x_gate, 5.45, 0.95, r"$g_v$")

    draw_vector_column(ax, x_final, y_vec, final_labels, color=COLORS["ink"], fontsize=12.0)

    # Two arrows link latent -> gate -> gated latent.
    arrow(ax, (9.98, 5.45), (10.82, 5.45), lw=1.2)
    arrow(ax, (12.28, 5.45), (13.12, 5.45), lw=1.2)

    add_top_label(ax, x_in + 0.45, "Predictor ($v$)\ninput")
    add_top_label(ax, x_h1, "Hidden\nLayer 1")
    add_top_label(ax, x_h2, "Hidden\nLayer 2")
    add_top_label(ax, x_lat, "Latent\noutput")
    add_top_label(ax, x_gate, "Seasonal\nGate")
    add_top_label(ax, x_final, "Gated\nlatent")

    draw_bottom_brackets(ax, x_in - 0.25, x_lat, x_final + 0.34)


    return fig


fig = draw_predictor_v_gated_latent_brackets()
fig.savefig(OUTPUT_BASENAME.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.005, facecolor=COLORS["bg"])
plt.close(fig)
print(OUTPUT_BASENAME.with_suffix(".pdf"))
