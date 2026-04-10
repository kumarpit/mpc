import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(__file__)
_FIELD_GRID = 3   # arrows per axis in force-field mode


def _box_faces(bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]


class Visualizer:
    def __init__(self, fans, r_free, r_actual, alpha_log, active_log,
                 pred_trajs, final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal):
        self.fans       = fans
        self.r_free     = r_free
        self.r_actual   = r_actual
        self.alpha_log  = alpha_log
        self.active_log = active_log
        self.pred_trajs = pred_trajs
        self.final_err  = final_err
        self.H          = H
        self.N          = N
        self.dt         = dt
        self.ALPHA_MAX  = ALPHA_MAX
        self.r0         = r0
        self.v0         = v0
        self.r_goal     = r_goal
        self.eff_alpha  = active_log * alpha_log
        self.total_act  = self.eff_alpha.sum(axis=1)
        self.t          = np.arange(len(alpha_log)) * dt

    def _draw_fans(self, ax, alpha_box=0.10, force_field=False):
        drawn = set()
        for fan in self.fans:
            key = tuple(fan.bounds.ravel())
            if key not in drawn:
                ax.add_collection3d(Poly3DCollection(
                    _box_faces(fan.bounds), alpha=alpha_box,
                    facecolor='#AAAAAA', edgecolor='#888888', linewidth=0.4,
                ))
                drawn.add(key)

            dims      = fan.bounds[:, 1] - fan.bounds[:, 0]
            arrow_len = dims.min() * 0.20
            d         = fan.direction * arrow_len

            if force_field:
                lo = fan.bounds[:, 0] + np.maximum(0.0, -d)
                hi = fan.bounds[:, 1] - np.maximum(0.0,  d)
                xs, ys, zs = [np.linspace(lo[i], hi[i], _FIELD_GRID) for i in range(3)]
                gx, gy, gz = np.meshgrid(xs, ys, zs)
                ax.quiver(gx.ravel(), gy.ravel(), gz.ravel(),
                          np.full(gx.size, d[0]),
                          np.full(gx.size, d[1]),
                          np.full(gx.size, d[2]),
                          color=fan.color, linewidth=0.8,
                          arrow_length_ratio=0.35, alpha=0.6)
            else:
                c = fan.bounds.mean(axis=1)
                ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                          color=fan.color, linewidth=1.8,
                          arrow_length_ratio=0.35, alpha=0.9)

    def _fig_overview(self):
        n_fans      = len(self.fans)
        ncols_right = max(1, int(np.ceil(np.sqrt(n_fans))))
        nrows_right = int(np.ceil(n_fans / ncols_right))

        fig = plt.figure(figsize=(14 + ncols_right * 3, max(8, nrows_right * 4)))
        gs  = GridSpec(nrows_right, 1 + ncols_right, figure=fig,
                       width_ratios=[3] + [1] * ncols_right,
                       hspace=0.48, wspace=0.38)

        ax3d     = fig.add_subplot(gs[:, 0], projection='3d')
        fan_axes = [fig.add_subplot(gs[i // ncols_right, i % ncols_right + 1])
                    for i in range(n_fans)]

        self._draw_fans(ax3d, force_field=True)
        ax3d.plot(*self.r_free.T, '--', color='#909090', linewidth=1.5, alpha=0.8)

        ax3d.plot(self.r_actual[:, 0], self.r_actual[:, 1], self.r_actual[:, 2],
                  color='steelblue', linewidth=3.0)

        ax3d.scatter(*self.r0,     s=140, c='lime',  marker='o', depthshade=False, zorder=6)
        ax3d.scatter(*self.r_goal, s=200, c='red',   marker='o', depthshade=False, zorder=6)
        ax3d.scatter(*self.r_goal, s=40,  c='white', marker='o', depthshade=False, zorder=7)
        v_scale = 0.4
        ax3d.quiver(*self.r0, *(self.v0 * v_scale),
                    color='black', linewidth=1.8, arrow_length_ratio=0.25)

        legend_els = [
            Line2D([0],[0], color='#909090',   linestyle='--', label='Free flight'),
            Line2D([0],[0], color='steelblue', linewidth=3,    label='MPC trajectory'),
            Line2D([0],[0], color='black',     linewidth=1.8,  label='Initial velocity'),
        ] + [Line2D([0],[0], color=f.color, linewidth=6, alpha=0.55, label=f.name) for f in self.fans]
        ax3d.legend(handles=legend_els, fontsize=7, loc='upper left', framealpha=0.75)
        ax3d.set_xlabel('X (m)', labelpad=6)
        ax3d.set_ylabel('Y (m)', labelpad=6)
        ax3d.set_zlabel('Z (m)', labelpad=6)
        ax3d.view_init(elev=20, azim=-50)

        for i, (fan, ax) in enumerate(zip(self.fans, fan_axes)):
            act = self.eff_alpha[:, i]
            ax.fill_between(self.t, act, color=fan.color, alpha=0.35)
            ax.plot(self.t, act, color=fan.color, linewidth=1.8)
            ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--')
            ax.set_ylim(-self.ALPHA_MAX * 1.1, self.ALPHA_MAX * 1.1)
            ax.set_title(fan.name, fontsize=8, color=fan.color, fontweight='bold')
            ax.set_xlabel('t (s)', fontsize=7)
            ax.set_ylabel('α (m/s²)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)

        return fig

    def _fig_per_step(self):
        ncols = 5
        nrows = int(np.ceil((self.N - 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'},
                                 figsize=(ncols * 4, nrows * 4))
        axes = np.array(axes).flatten()

        for k, (traj, ax) in enumerate(zip(self.pred_trajs, axes)):
            self._draw_fans(ax, alpha_box=0.07)
            if k > 0:
                ax.plot(self.r_actual[:k+1, 0], self.r_actual[:k+1, 1],
                        self.r_actual[:k+1, 2], color='lime', linewidth=2.0)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='steelblue', linewidth=1.8, linestyle='--')
            ax.scatter(*self.r0,          s=60,  c='lime',   marker='o', depthshade=False)
            ax.scatter(*self.r_actual[k], s=80,  c='yellow', marker='o', depthshade=False, zorder=6)
            ax.scatter(*self.r_goal,      s=120, c='red',    marker='o', depthshade=False)
            ax.set_title(f'k={k}  H={min(self.H, self.N - k)}', fontsize=9, fontweight='bold')
            ax.set_xlabel('X', fontsize=6, labelpad=1)
            ax.set_ylabel('Y', fontsize=6, labelpad=1)
            ax.set_zlabel('Z', fontsize=6, labelpad=1)
            ax.tick_params(labelsize=5)
            ax.view_init(elev=20, azim=-50)

        for ax in axes[self.N-1:]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    def show(self):
        fig1 = self._fig_overview()
        fig3 = self._fig_per_step()

        plt.show()

        for fig, name in [(fig1, 'poc7_visualization.png'),
                          (fig3, 'poc7_per_step.png')]:
            fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
            print(f"Saved: {name}")


def visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
              final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal):
    Visualizer(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
               final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal).show()
