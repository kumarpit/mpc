"""
board_diagram.py  –  3-D diagram of a paper-toss level.

Usage:
    python diagrams/board_diagram.py levels/level2.toml
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from level import load_level

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ── geometry helpers ──────────────────────────────────────────────────────────

def box_faces(bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def draw_board(level_path: str):
    lv = load_level(level_path)
    level_name = os.path.splitext(os.path.basename(level_path))[0]

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#FFFFFF")

    # ── fan zones ─────────────────────────────────────────────────────────────
    GRID = 4        # arrow grid points per axis
    legend_handles = []
    for fan in lv.fans:
        faces = box_faces(fan.bounds)
        poly = Poly3DCollection(
            faces,
            alpha=0.10,
            facecolor=fan.color,
            edgecolor=fan.color,
            linewidth=0.6,
        )
        ax.add_collection3d(poly)

        # force-field arrow grid
        dims = fan.bounds[:, 1] - fan.bounds[:, 0]   # [wx, wy, wz]
        arrow_len = dims.min() * 0.20
        d = fan.direction * arrow_len

        # shrink grid range so arrow tips stay inside the bounding box
        lo = fan.bounds[:, 0] + np.maximum(0.0, -d)
        hi = fan.bounds[:, 1] - np.maximum(0.0,  d)

        xs = np.linspace(lo[0], hi[0], GRID)
        ys = np.linspace(lo[1], hi[1], GRID)
        zs = np.linspace(lo[2], hi[2], GRID)
        gx, gy, gz = np.meshgrid(xs, ys, zs)

        ax.quiver(
            gx.ravel(), gy.ravel(), gz.ravel(),
            np.full(gx.size, d[0]),
            np.full(gx.size, d[1]),
            np.full(gx.size, d[2]),
            color=fan.color,
            linewidth=0.9,
            arrow_length_ratio=0.35,
            alpha=0.65,
        )

        legend_handles.append(
            mpatches.Patch(facecolor=fan.color, edgecolor=fan.color,
                           alpha=0.55, label=fan.name)
        )

    # ── start marker ─────────────────────────────────────────────────────────
    ax.scatter(*lv.r0, s=90, c="#27AE60", edgecolors="white",
               linewidths=1.4, zorder=6, depthshade=False)
    ax.text(lv.r0[0], lv.r0[1], lv.r0[2] + 0.3, "Start",
            fontsize=9, color="#27AE60", fontweight="bold")

    # ── goal marker (filled circle + ring) ───────────────────────────────────
    ax.scatter(*lv.r_goal, s=160, c="#E74C3C", edgecolors="white",
               linewidths=2.0, zorder=6, depthshade=False)
    ax.scatter(*lv.r_goal, s=30, c="white", zorder=7, depthshade=False)
    ax.text(lv.r_goal[0], lv.r_goal[1], lv.r_goal[2] + 0.35, "Goal",
            fontsize=9, color="#E74C3C", fontweight="bold")

    # ── axes labels / style ───────────────────────────────────────────────────
    ax.set_xlabel("X (m)", labelpad=8, fontsize=10)
    ax.set_ylabel("Y (m)", labelpad=8, fontsize=10)
    ax.set_zlabel("Z (m)", labelpad=8, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#DDDDDD")
    ax.yaxis.pane.set_edgecolor("#DDDDDD")
    ax.zaxis.pane.set_edgecolor("#DDDDDD")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD")
    ax.view_init(elev=22, azim=-55)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles += [
        mpatches.Patch(color="#27AE60", label="Start position"),
        mpatches.Patch(color="#E74C3C", label="Goal position"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
              loc="upper left", borderpad=0.8)

    fig.tight_layout()
    plt.show()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{level_name}_board.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a 3-D level diagram")
    parser.add_argument("level", help="Path to a level TOML file")
    args = parser.parse_args()
    draw_board(args.level)
