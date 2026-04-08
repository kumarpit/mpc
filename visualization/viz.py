import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import os

OUT_DIR = os.path.dirname(__file__)


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


def draw_fans(ax, fans, alpha_box=0.10, alpha_arrow=1.0):
    for fan in fans:
        poly = Poly3DCollection(box_faces(fan.bounds), alpha=alpha_box,
                                facecolor=fan.color, edgecolor=fan.color,
                                linewidth=0.4)
        ax.add_collection3d(poly)
        c = fan.bounds.mean(axis=1)
        d = fan.direction * 0.6
        ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                  color=fan.color, linewidth=1.8, arrow_length_ratio=0.35,
                  alpha=alpha_arrow)


def visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
              final_err, H, N, dt, ALPHA_MAX, r0, r_goal):
    eff_alpha = active_log * alpha_log
    total_act = eff_alpha.sum(axis=1)
    t = np.arange(len(alpha_log)) * dt

    # ── Figure 1: 3D + 3×3 fan panels ────────────────────────────────────────
    fig1 = plt.figure(figsize=(22, 12))
    gs1  = GridSpec(3, 4, figure=fig1, width_ratios=[3, 1, 1, 1],
                    hspace=0.48, wspace=0.38)

    ax3d     = fig1.add_subplot(gs1[:, 0], projection='3d')
    fan_axes = [fig1.add_subplot(gs1[i // 3, i % 3 + 1]) for i in range(9)]

    draw_fans(ax3d, fans)
    ax3d.plot(*r_free.T, '--', color='#909090', linewidth=1.5, alpha=0.8,
              label='Free flight')

    n_pred = len(pred_trajs)
    for i, traj in enumerate(pred_trajs):
        fade = 0.06 + 0.30 * (i / max(n_pred - 1, 1))
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color='steelblue', linewidth=0.9, alpha=fade, linestyle=':')

    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=max(float(total_act.max()), 0.1))
    for k in range(len(r_actual) - 1):
        c = cmap(norm(total_act[k]))
        ax3d.plot(r_actual[k:k+2, 0], r_actual[k:k+2, 1], r_actual[k:k+2, 2],
                  color=c, linewidth=3.0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=ax3d, shrink=0.42, pad=0.08,
                       label='Fan effort applied (m s⁻²)')
    cb.ax.tick_params(labelsize=8)

    ax3d.scatter(*r0,     s=140, c='lime', marker='o', depthshade=False, zorder=6)
    ax3d.scatter(*r_goal, s=240, c='red',  marker='*', depthshade=False, zorder=6)
    ax3d.text(r0[0]-0.2, r0[1]-0.5, r0[2]+0.2,  'Start',
              fontsize=9, color='lime', fontweight='bold')
    ax3d.text(r_goal[0]+0.2, r_goal[1]+0.1, r_goal[2]+0.3, 'Goal',
              fontsize=9, color='red', fontweight='bold')

    legend_els = [
        Line2D([0],[0], color='#909090', linestyle='--', label='Free flight'),
        Line2D([0],[0], color='steelblue', linewidth=0.9, linestyle=':', alpha=0.5,
               label=f'H={H} predicted plans'),
        Line2D([0],[0], color=cmap(0.85), linewidth=3, label='MPC trajectory'),
    ] + [Line2D([0],[0], color=f.color, linewidth=6, alpha=0.55, label=f.name)
         for f in fans]
    ax3d.legend(handles=legend_els, fontsize=7, loc='upper left', framealpha=0.75)
    ax3d.set_xlabel('X (m)', labelpad=6)
    ax3d.set_ylabel('Y (m)', labelpad=6)
    ax3d.set_zlabel('Z (m)', labelpad=6)
    ax3d.set_title(f'3D Fan-Guided Paper Toss\n(MPC, Fixed H={H} Horizon, 9 Fans)',
                   fontsize=11, fontweight='bold', pad=12)
    ax3d.view_init(elev=20, azim=-50)

    for i, (fan, ax) in enumerate(zip(fans, fan_axes)):
        act = eff_alpha[:, i]
        ax.fill_between(t, act, color=fan.color, alpha=0.35)
        ax.plot(t, act, color=fan.color, linewidth=1.8)
        ax.set_ylim(0, ALPHA_MAX * 1.1)
        ax.set_title(fan.name, fontsize=8, color=fan.color, fontweight='bold')
        ax.set_xlabel('t (s)', fontsize=7)
        ax.set_ylabel('α (m/s²)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

    fig1.suptitle(
        f'Paper Toss — MPC Fixed H={H}  |  terminal error: {final_err:.4f} m',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(os.path.join(OUT_DIR, 'poc7_visualization.png'), dpi=150, bbox_inches='tight')
    print("Saved: poc7_visualization.png")

    # ── Figure 2: overlay + total ─────────────────────────────────────────────
    fig2, (ax_overlay, ax_total) = plt.subplots(1, 2, figsize=(11, 4))

    for i, fan in enumerate(fans):
        ax_overlay.plot(t, eff_alpha[:, i], color=fan.color,
                        linewidth=1.6, label=fan.name.split('(')[0].strip())
    ax_overlay.set_title('All fans (overlay)', fontsize=10, fontweight='bold')
    ax_overlay.set_xlabel('t (s)'); ax_overlay.set_ylabel('α (m/s²)')
    ax_overlay.legend(fontsize=7, framealpha=0.7, ncol=2)
    ax_overlay.tick_params(labelsize=8); ax_overlay.grid(alpha=0.3)

    ax_total.fill_between(t, total_act, color='#6A3FA0', alpha=0.35)
    ax_total.plot(t, total_act, color='#6A3FA0', linewidth=1.8)
    ax_total.set_title('Total fan effort', fontsize=10, fontweight='bold')
    ax_total.set_xlabel('t (s)'); ax_total.set_ylabel('Σα (m/s²)')
    ax_total.tick_params(labelsize=8); ax_total.grid(alpha=0.3)

    fig2.suptitle(f'Fan Effort Summary (H={H})', fontsize=12, fontweight='bold')
    fig2.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'poc7_effort.png'), dpi=150, bbox_inches='tight')
    print("Saved: poc7_effort.png")

    # ── Figure 3: per-timestep predicted plans ────────────────────────────────
    ncols = 5
    nrows = int(np.ceil((N - 1) / ncols))
    fig3, axes3 = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'},
                                figsize=(ncols * 4, nrows * 4))
    axes3 = np.array(axes3).flatten()

    for k, (traj, ax) in enumerate(zip(pred_trajs, axes3)):
        draw_fans(ax, fans, alpha_box=0.07, alpha_arrow=0.5)

        if k > 0:
            ax.plot(r_actual[:k+1, 0], r_actual[:k+1, 1],
                    r_actual[:k+1, 2], color='lime', linewidth=2.0)

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color='steelblue', linewidth=1.8, linestyle='--')

        ax.scatter(*r0,           s=60,  c='lime',   marker='o', depthshade=False)
        ax.scatter(*r_actual[k],  s=80,  c='yellow', marker='o', depthshade=False, zorder=6)
        ax.scatter(*r_goal,       s=120, c='red',    marker='*', depthshade=False)

        horizon_k = min(H, N - k)
        ax.set_title(f'k={k}  H={horizon_k}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X', fontsize=6, labelpad=1)
        ax.set_ylabel('Y', fontsize=6, labelpad=1)
        ax.set_zlabel('Z', fontsize=6, labelpad=1)
        ax.tick_params(labelsize=5)
        ax.view_init(elev=20, azim=-50)

    for ax in axes3[N-1:]:
        ax.set_visible(False)

    fig3.suptitle(
        f'MPC H={H}: Predicted Plan at Each Timestep\n'
        '(yellow=current, blue dashed=5-step QP plan, green=actual path so far)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'poc7_per_step.png'), dpi=120, bbox_inches='tight')
    print("Saved: poc7_per_step.png")

    plt.show()
