#!/usr/bin/env python
# coding: utf-8
"""Figure 4: Argo-GNN schematic in three steps.

A single composition that fuses: (1) a realistic 3-D Argo scatter clustered
into five compartments, (2) a slab with the schematic 5-node graph and
message-passing edges, and (3) the concatenated node-embedding ribbon.
The five cluster colours are the visual thread that links the three steps.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Recreate paper Figure 4: Argo-GNN diagram")
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

# Cluster palette: five monotone tones from the lightest (c1) to the darkest (c5)
# plus auxiliary ink/muted/background entries used by the schematic frame.
COLORS = {
    "ink": "#111111",
    "muted": "#6e7681",
    "bg": "#ffffff",
    "point": "#3da5e8",
    "c1": "#DCE5BC",   # beige
    "c2": "#AEC3B0",   # ash grey
    "c3": "#598392",   # air force blue
    "c4": "#124559",   # dark teal
    "c5": "#01161E",   # ink black
}

ARGO_Q_PATH = REPO_ROOT / "data_preparation/data/input/virtual_argo/oras5_virtual_argo_density_input_Q_34p5S_pm3deg_1958_2025.nc"
OUTPUT_BASENAME = OUTPUT_DIR / "fig04_argo_gnn"
print(ARGO_Q_PATH)
print(OUTPUT_BASENAME)




# --- Load one quarterly Argo sample and cluster its profiles into 5 compartments ---

ds = xr.open_dataset(ARGO_Q_PATH)
profile_mask = ds["profile_mask"].values.astype(bool)
sample_idx = int(np.where(ds["sample"].values == np.datetime64("2018-02-15"))[0][0])
valid = profile_mask[sample_idx]

lat = ds["lat"].values[sample_idx, valid].astype(np.float64)
lon = ds["lon"].values[sample_idx, valid].astype(np.float64)
month_in_quarter = ds["month_in_quarter"].values[sample_idx, valid].astype(np.int64)
month_offset = month_in_quarter - 1       # map 0/1/2 to -1/0/+1 around the central month

# KMeans with 5 clusters on standardised (lat, lon, month) coordinates.
coords = np.column_stack([lat, lon, month_offset.astype(np.float64)])
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
kmeans = KMeans(n_clusters=5, n_init=20, random_state=7)
labels = kmeans.fit_predict(coords_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

node_colors = [COLORS[k] for k in ("c1", "c2", "c3", "c4", "c5")]
cluster_colors = np.array(node_colors, dtype=object)
point_colors = cluster_colors[labels]

# Scatter-only colour swap: exchange c2 and c5 so the visible darkest centroid
# in the 3-D cube does not land on the same spot as the darkest node in the slabs.
scatter_cluster_colors = cluster_colors.copy()
scatter_cluster_colors[[1, 4]] = scatter_cluster_colors[[4, 1]]
scatter_point_colors = scatter_cluster_colors[labels]

# Nudge the two central centroids apart so the 3-D rendering reads cleanly.
display_centroids = centroids.copy()
display_centroids[3] += np.array([0.28, 3.2, 0.14])
display_centroids[4] += np.array([-0.28, -3.2, -0.14])

lat_bounds = (-37.5, -31.5)
lon_bounds = (-51.5, 17.56)


def _normalise(values, inset=0.16):
    """Map values to [inset, 1 - inset] so nodes stay inset-away from the slab borders."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5)
    return inset + (1.0 - 2.0 * inset) * (values - lo) / (hi - lo)


# Place the five centroids on the slab's receding face: longitude -> horizontal (depth), latitude -> vertical.
node_u = _normalise(centroids[:, 1])
node_v = _normalise(centroids[:, 0])
node_uv = np.column_stack([node_u, node_v])

# Visual nudge for node c3 (index 2) so it stops overlapping the (c4, c5) edge.
node_uv[2, 0] = max(0.03, node_uv[2, 0] - 0.10)

# Schematic edge list, identical across the two slabs so the two drawings read as one graph.
EDGES = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]

print("centroids (lat, lon, m_off):")
print(centroids.round(2))
print("normalised node positions on slab face (u, v):")
print(node_uv.round(3))




# --- Geometry and style of the oblique-projection slabs ---

# One slab is a thin vertical tile seen in oblique projection: a narrow front face
# on the left, a top parallelogram, and a large receding face that hosts the graph.
FACE_W = 0.13          # width of the narrow front face (slab thickness)
SLAB_H = 2.50          # vertical slab height
DEPTH_DX = 1.35        # horizontal projection of the depth vector
DEPTH_DY = 0.90        # vertical projection of the depth vector (larger = more tilt)

FACE_FC = "#dfe3e8"    # narrow front face fill
TOP_FC = "#eef1f4"     # top parallelogram fill
GRAPH_FC = "#fcfcfc"   # large receding face fill (white)
SLAB_EC = "#9aa3ad"    # slab edges
SLAB_LW = 0.9

# Graph edge and node style (monochromatic palette).
EDGE_COLOR = "#8a93a0"
EDGE_LW = 1.05
NODE_SIZE = 96
NODE_EC = "#1f2933"
NODE_LW = 0.9

# Message dots that travel along the Cheb-Conv edges in the message-passing slab.
MESSAGE_DOT_SIZE = 24
MESSAGE_DOT_LW = 0.35
MESSAGE_T_NEAR = 0.30
MESSAGE_T_FAR = 0.70

# Concatenated-embedding ribbon style.
RIBBON_W = 0.45
RIBBON_LW = 0.9

ARROW_COLOR = COLORS["ink"]
ARROW_LW = 1.1

# Vertical gap between each block and its title.
TITLE_GAP = 0.32


def draw_slab(ax, x0, y0, title, title_y, face_color=GRAPH_FC):
    """Draw one oblique-projection slab and return a (u, v) -> (x, y) projection onto its receding face."""
    # Front, top and receding face polygons.
    fbl = (x0, y0)
    fbr = (x0 + FACE_W, y0)
    ftr = (x0 + FACE_W, y0 + SLAB_H)
    ftl = (x0, y0 + SLAB_H)
    bbr = (fbr[0] + DEPTH_DX, fbr[1] + DEPTH_DY)
    btr = (ftr[0] + DEPTH_DX, ftr[1] + DEPTH_DY)
    btl = (ftl[0] + DEPTH_DX, ftl[1] + DEPTH_DY)

    front = Polygon([fbl, fbr, ftr, ftl], closed=True,
                    facecolor=FACE_FC, edgecolor=SLAB_EC, linewidth=SLAB_LW, zorder=1)
    top = Polygon([ftl, ftr, btr, btl], closed=True,
                  facecolor=TOP_FC, edgecolor=SLAB_EC, linewidth=SLAB_LW, zorder=2)
    graph_face = Polygon([fbr, bbr, btr, ftr], closed=True,
                         facecolor=face_color, edgecolor=SLAB_EC, linewidth=SLAB_LW, zorder=3)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(graph_face)

    title_x = x0 + FACE_W + DEPTH_DX / 2
    ax.text(title_x, title_y, title,
            ha="center", va="bottom", fontsize=14, color=COLORS["ink"],
            linespacing=1.05)

    def project(u, v):
        """Project a (u in [0, 1], v in [0, 1]) coordinate onto the receding face."""
        bx = x0 + FACE_W
        by = y0
        return bx + u * DEPTH_DX, by + u * DEPTH_DY + v * SLAB_H

    return project, title_x


def draw_graph_on_face(ax, project, with_messages=False):
    """Draw the 5-node graph on the slab's receding face, optionally with travelling message dots."""
    xs, ys = zip(*[project(u, v) for u, v in node_uv])
    for i, j in EDGES:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color=EDGE_COLOR, linewidth=EDGE_LW,
                solid_capstyle="round", zorder=4)
        if with_messages:
            # Two coloured dots per edge, approaching the opposite endpoints.
            dx, dy = xs[j] - xs[i], ys[j] - ys[i]
            mx_near = xs[i] + MESSAGE_T_NEAR * dx
            my_near = ys[i] + MESSAGE_T_NEAR * dy
            mx_far = xs[i] + MESSAGE_T_FAR * dx
            my_far = ys[i] + MESSAGE_T_FAR * dy
            ax.scatter([mx_near], [my_near], s=MESSAGE_DOT_SIZE,
                       c=[node_colors[j]], edgecolors=NODE_EC,
                       linewidths=MESSAGE_DOT_LW, zorder=4.5)
            ax.scatter([mx_far], [my_far], s=MESSAGE_DOT_SIZE,
                       c=[node_colors[i]], edgecolors=NODE_EC,
                       linewidths=MESSAGE_DOT_LW, zorder=4.5)
    ax.scatter(xs, ys, s=NODE_SIZE, c=node_colors,
               edgecolors=NODE_EC, linewidths=NODE_LW, zorder=5)


def draw_ribbon(ax, x0, y0, title_y, title, n_segments=5):
    """Draw the final concatenated-embedding ribbon from the lightest to the darkest tone."""
    w = RIBBON_W
    h = 0.75 * (SLAB_H + DEPTH_DY)
    seg_h = h / n_segments

    for k in range(n_segments):
        bot_y = y0 + h - (k + 1) * seg_h       # draw top-down: lightest at the top
        seg = Rectangle((x0, bot_y), w, seg_h,
                        facecolor=node_colors[k],
                        edgecolor="none", zorder=2)
        ax.add_patch(seg)

    outer = Rectangle((x0, y0), w, h,
                      facecolor="none", edgecolor=SLAB_EC,
                      linewidth=RIBBON_LW, zorder=4)
    ax.add_patch(outer)

    cx = x0 + w / 2
    ax.text(cx, title_y, title,
            ha="center", va="bottom", fontsize=14,
            color=COLORS["ink"], linespacing=1.05)

    return x0, y0, w, h, cx


def draw_arrow(ax, x_from, x_to, y):
    """Draw a thin horizontal arrow between two schematic blocks."""
    arrow = FancyArrowPatch((x_from, y), (x_to, y),
                            arrowstyle="-|>", mutation_scale=12,
                            color=ARROW_COLOR, linewidth=ARROW_LW,
                            shrinkA=0, shrinkB=0, zorder=6)
    ax.add_patch(arrow)


def draw_scatter_on_ax(ax3d):
    """Render the 3-D Argo scatter with its five cluster centroids into the given 3-D axes."""
    ax3d.scatter(lat, lon, month_offset,
                 s=9, c=scatter_point_colors, alpha=0.58,
                 depthshade=False, linewidths=0.0)
    ax3d.scatter(
        display_centroids[:, 0], display_centroids[:, 1], display_centroids[:, 2],
        s=320, c=scatter_cluster_colors, edgecolors=NODE_EC, linewidths=0.5,
        depthshade=False, zorder=5,
    )
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_box_aspect((0.80, 1.34, 0.72))
    ax3d.set_xlim(lat_bounds)
    ax3d.set_ylim(lon_bounds)
    ax3d.set_zlim(-1.2, 1.2)
    ax3d.set_xlabel("Latitude", labelpad=2, fontsize=11)
    ax3d.set_ylabel("Longitude", labelpad=7, fontsize=11)
    ax3d.set_zlabel("")
    ax3d.set_xticks([-37.5, -34.5, -31.5])
    ax3d.set_xticklabels(["-31.5", "-34.5", ""])
    ax3d.set_yticks([-50, -30, -10, 10])
    ax3d.set_zticks([-1, 0, 1])
    ax3d.set_zticklabels(["-1", "0", "+1"])
    ax3d.tick_params(axis="both", which="major", pad=0, labelsize=10)
    ax3d.grid(True, alpha=0.22)
    # Hide the default coloured 3-D panes.
    ax3d.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax3d.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax3d.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax3d.xaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
    ax3d.yaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
    ax3d.zaxis.pane.set_edgecolor((0.88, 0.88, 0.88, 1.0))
    ax3d.xaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.65)
    ax3d.yaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.65)
    ax3d.zaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.65)
    ax3d.xaxis._axinfo["grid"]["linewidth"] = 0.8
    ax3d.yaxis._axinfo["grid"]["linewidth"] = 0.8
    ax3d.zaxis._axinfo["grid"]["linewidth"] = 0.8
    ax3d.text(-31.82, lon_bounds[0] - 2.6, -1.18, "-37.5",
              fontsize=10, ha="right", va="top", color=COLORS["ink"])
    ax3d.text2D(1.00, 0.58, "Month offset", transform=ax3d.transAxes,
                rotation=90, fontsize=11, ha="left", va="center",
                color=COLORS["ink"])


def draw_panel_b_architecture():
    """Compose the three-step panel: (1) compartment assignment, (2) message passing, (3) node concatenation."""
    y0 = 0.35
    face_mid_y = y0 + SLAB_H / 2 + DEPTH_DY / 2
    block_top_y = y0 + SLAB_H + DEPTH_DY
    slab_title_y = block_top_y + TITLE_GAP

    visual_gap = 0.26
    arrow_length = 0.45
    slab_total_w = FACE_W + DEPTH_DX

    # Matplotlib's 3-D axes reserve margins for perspective, so the scatter axes box
    # is slightly larger than the visible cube so it matches the slab/ribbon heights.
    scatter_axis_scale = 1.20
    scatter_base_w = 4.0
    scatter_base_h = block_top_y - y0
    scatter_center_x = -2.5
    scatter_y_top = block_top_y + 0.48
    scatter_y_bot = scatter_y_top - scatter_base_h * scatter_axis_scale
    scatter_x_left = scatter_center_x - scatter_base_w * scatter_axis_scale / 2
    scatter_x_right = scatter_center_x + scatter_base_w * scatter_axis_scale / 2
    # Visible right edge of the scatter block (ignores the invisible right padding).
    scatter_visual_right = -0.35

    inter_block_w = 2 * visual_gap + arrow_length
    x_msg = scatter_visual_right + inter_block_w
    x_ribbon = x_msg + slab_total_w + inter_block_w

    x_left_bound = scatter_x_left + 0.30
    x_right_bound = x_ribbon + RIBBON_W + 0.25
    y_bottom = 0.24
    y_top = slab_title_y + 0.31
    total_w = x_right_bound - x_left_bound
    total_h = y_top - y_bottom

    # Keep the original data-unit-to-inch scale so this three-block layout keeps
    # the same visual proportions as the original four-block one.
    reference_x_ribbon = scatter_visual_right + inter_block_w + 2 * (slab_total_w + inter_block_w)
    reference_right_bound = reference_x_ribbon + RIBBON_W + 0.4
    reference_total_w = reference_right_bound - x_left_bound
    fig_scale = 11.4 / reference_total_w
    fig = plt.figure(figsize=(total_w * fig_scale, total_h * fig_scale), facecolor=COLORS["bg"])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("none")
    ax.set_zorder(2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(x_left_bound, x_right_bound)
    ax.set_ylim(y_bottom, y_top)

    # 3-D scatter axes, positioned in figure coordinates from the data-space zone above.
    fig_left = (scatter_x_left - x_left_bound) / total_w
    fig_bottom = (scatter_y_bot - y_bottom) / total_h
    fig_width = (scatter_x_right - scatter_x_left) / total_w
    fig_height = (scatter_y_top - scatter_y_bot) / total_h
    scatter_ax = fig.add_axes(
        [fig_left, fig_bottom, fig_width, fig_height],
        projection="3d",
    )
    scatter_ax.set_facecolor(COLORS["bg"])
    scatter_ax.set_zorder(1)
    draw_scatter_on_ax(scatter_ax)

    # Step (1) title above the scatter.
    scatter_center_x = (scatter_x_left + scatter_x_right) / 2
    ax.text(scatter_center_x, slab_title_y, "(1) Compartment\nassignment",
            ha="center", va="bottom", fontsize=14, color=COLORS["ink"],
            linespacing=1.05, zorder=20)

    # Step (2): message-passing slab with its 5-node graph.
    project_msg, _ = draw_slab(
        ax, x_msg, y0, "(2) Message\npassing", slab_title_y,
    )

    # Step (3): concatenated-embedding ribbon.
    ribbon_h = 0.75 * (SLAB_H + DEPTH_DY)
    ribbon_y0 = face_mid_y - ribbon_h / 2
    ribbon_x0, _, ribbon_w_real, _, _ = draw_ribbon(
        ax, x_ribbon, ribbon_y0, slab_title_y, "(3) Node\nconcatenation",
    )

    draw_graph_on_face(ax, project_msg, with_messages=True)

    # Two arrows linking the three blocks.
    a1_start = scatter_visual_right + visual_gap
    a1_end = a1_start + arrow_length
    a2_start = x_msg + slab_total_w + visual_gap
    a2_end = a2_start + arrow_length
    draw_arrow(ax, a1_start, a1_end, face_mid_y)
    draw_arrow(ax, a2_start, a2_end, face_mid_y)

    return fig


fig = draw_panel_b_architecture()
fig.savefig(OUTPUT_BASENAME.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01, facecolor=COLORS["bg"])
plt.close(fig)
print(OUTPUT_BASENAME.with_suffix(".pdf"))
