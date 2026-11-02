"""Simple live Matplotlib UI and static dashboard renderer.

Display code consumes VisualizationFrame only: no hidden simulator state,
no planner imports, and no need to modify the mission contract.
"""
from __future__ import annotations
from pathlib import Path
from .frame import VisualizationFrame


def interactive_backend_available() -> bool:
    import matplotlib
    backend = matplotlib.get_backend().lower()
    # GUI names include 'agg' (TkAgg, QtAgg), so never reject by substring.
    return backend.startswith(('macosx', 'qt', 'tk', 'gtk', 'wx', 'webagg', 'nbagg'))


class DashboardRenderer:
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, (self.ax_map, self.ax_panel) = plt.subplots(
            1, 2, figsize=(13.4, 7.6), gridspec_kw={'width_ratios': [3.3, 1.5]})
        self.fig.patch.set_facecolor('#fcfcfd')

    def draw(self, frame: VisualizationFrame):
        import matplotlib.colors as colors
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from textwrap import shorten
        ax, panel = self.ax_map, self.ax_panel
        ax.clear()
        panel.clear()
        grid = frame.occupancy
        palette = colors.ListedColormap(['#cfd5dd', '#f5f7fa', '#273444'])
        image = [[cell+1 for cell in row] for row in grid]
        x0, y0 = frame.origin_x_m, frame.origin_y_m
        x1 = x0 + len(grid[0])*frame.cell_size_m
        y1 = y0 + len(grid)*frame.cell_size_m
        ax.imshow(image, origin='lower', interpolation='nearest', cmap=palette,
                  vmin=-0.5, vmax=2.5,
                  extent=[x0, x1, y0, y1], zorder=0)
        ros_mode = frame.planner is not None and frame.planner.mode == 'ROS_BRIDGE'
        if ros_mode:
            left = min(frame.home.x, frame.drone.x)-7
            right = max(frame.home.x, frame.drone.x)+7
            bottom_view = min(frame.home.y,frame.drone.y)-7
            top_view = max(frame.home.y,frame.drone.y)+7
            ax.set_xlim(max(x0,left), min(x1,right))
            ax.set_ylim(max(y0,bottom_view), min(y1,top_view))
        else:
            ax.set_xlim(x0,x1)
            ax.set_ylim(y0,y1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Map X (m)')
        ax.set_ylabel('Map Y (m)')
        title = ('LiDAR XY DEBUG PROJECTION (NOT NAVIGABLE)'
                 if frame.planner and frame.planner.mode == 'ROS_BRIDGE'
                 else 'Observed occupancy')
        ax.set_title(f'{title}   |   map v{frame.map_version}',
                     loc='left', fontsize=12, fontweight='bold')
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(color='#94a3b8', lw=0.5, alpha=0.24)

        if frame.path:
            ax.plot([pt.x for pt in frame.path], [pt.y for pt in frame.path],
                    color='#0e7490', lw=2.1, alpha=.85, zorder=3)
        if frame.frontiers:
            ax.scatter([f.position.x for f in frame.frontiers],
                       [f.position.y for f in frame.frontiers],
                       color='#d97706', marker='s', s=45, edgecolors='white',
                       linewidths=0.8, zorder=4)
            for f in frame.frontiers[:5]:
                ax.annotate(f'{f.information_gain:.0f}',
                            (f.position.x, f.position.y), xytext=(4, 4),
                            textcoords='offset points', fontsize=7, color='#92400e')
        for entity in frame.discovered_entities:
            ax.scatter([entity.position.x], [entity.position.y], color='#dc2626',
                       marker='*', s=180, edgecolors='white', linewidths=.6, zorder=5)
            ax.annotate(entity.class_id, (entity.position.x, entity.position.y),
                        xytext=(5, -12), textcoords='offset points', fontsize=8,
                        color='#991b1b', zorder=5)
        ax.scatter([frame.home.x], [frame.home.y], marker='P', color='#059669',
                   s=135, edgecolors='white', linewidths=.6, zorder=6)
        ax.scatter([frame.drone.x], [frame.drone.y], marker='o', color='#7c3aed',
                   s=140, edgecolors='white', linewidths=.8, zorder=7)
        legend = [
            Patch(facecolor='#cfd5dd', label='Unknown'),
            Patch(facecolor='#f5f7fa', edgecolor='#94a3b8', label='Free'),
            Patch(facecolor='#273444', label='Occupied'),
            Line2D([], [], color='#0e7490', linewidth=2, label='Executed path'),
            Line2D([], [], marker='s', color='#d97706', ls='', label='Frontier'),
            Line2D([], [], marker='*', color='#dc2626', ls='', label='Discovered'),
            Line2D([], [], marker='P', color='#059669', ls='', label='Home'),
            Line2D([], [], marker='o', color='#7c3aed', ls='', label='Drone'),
        ]
        ax.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5,-.25),
                  ncol=4, fontsize=8, frameon=False)

        panel.axis('off')
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        lines = [
            ('MISSION', '#0f172a', 0.98, 11, 'bold'),
            (shorten(frame.mission_id, 28), '#334155', 0.93, 10, 'normal'),
            (f'Step {frame.state_seq}   •   {frame.coverage:.0%} observed',
             '#475569', 0.865, 10, 'normal'),
            ('LAST ACTION', '#0f172a', .79, 10, 'bold'),
            (shorten(frame.selected_action_id or '(initial scan)', 29), '#2563eb', .745, 9, 'normal'),
            ('MISSION GOALS', '#0f172a', .67, 10, 'bold'),
        ]
        goal_y = 0.623
        for goal, status in frame.goals.items():
            color = {'VERIFIED':'#059669', 'FAILED':'#dc2626',
                     'PENDING':'#b45309', 'UNKNOWN':'#64748b'}.get(status, '#475569')
            lines.append((shorten(goal, 18) + '  ·  ' + status, color, goal_y, 9, 'normal'))
            goal_y -= 0.046
        bottom = min(.445, goal_y - .01)
        resources_known = not ros_mode
        if resources_known:
            lines.extend([
            ('FLIGHT RESOURCES', '#0f172a', bottom, 10, 'bold'),
            (f'Energy  {frame.remaining_energy_wh:.1f} Wh', '#334155', bottom-.05, 9, 'normal'),
            (f'Return est.  {frame.return_energy_wh:.1f} Wh', '#334155', bottom-.095, 9, 'normal'),
            (f'Reserve  {frame.reserve_energy_wh:.1f} Wh', '#334155', bottom-.14, 9, 'normal'),
            (f'Time  {frame.remaining_time_s:.0f} s', '#334155', bottom-.185, 9, 'normal'),
            (f'Frontiers {len(frame.frontiers)}  •  Actions {frame.candidate_action_count}',
             '#475569', bottom-.23, 9, 'normal'),
        ])
        else:
            lines.extend([('ROS2 INPUTS ONLY', '#0f172a', bottom, 10, 'bold'),
                          ('Battery, return and safe routes: UNKNOWN', '#b45309', bottom-.06, 8, 'normal'),
                          ('LiDAR projection is not 3D flight space', '#b45309', bottom-.105, 8, 'normal')])
        if frame.planner is not None:
            pf = frame.planner
            lines.extend([
                ('ROS INPUT STATUS' if ros_mode else 'HIGH-LEVEL PLANNER', '#0f172a', .180, 9, 'bold'),
                (shorten(f'{pf.mode} | {pf.model or "baseline"} | {pf.status}', 51), '#1d4ed8', .153, 8, 'normal'),
                (shorten((pf.rationale or 'No action proposed yet'), 57), '#475569', .129, 7.5, 'normal'),
                (f'Latency: {pf.latency_s:.2f}s' if pf.latency_s is not None else 'Latency: n/a', '#64748b', .105, 8, 'normal'),
            ])
        if frame.safety is not None:
            sf = frame.safety
            base = .075 if frame.planner is not None else .128
            lines.extend([
                ('SAFETY SUPERVISOR', '#0f172a', base, 9, 'bold'),
                (shorten(f'{sf.mode} | return route: {"YES" if sf.return_route_known else "NO"} | safe: {sf.feasible_optional_count}', 51),
                 '#047857' if sf.mode in ('NORMAL','HOME') else '#b45309', base-.027, 8, 'normal'),
                (shorten((sf.last_status or 'NO_EVENT') + ': ' + (sf.last_reason or '-'), 48),
                 '#b91c1c' if sf.last_status in ('REJECTED','CONTINGENCY_LAND') else '#475569',
                 base-.052, 7.5, 'normal'),
            ])
        for label, color, y, fontsize, weight in lines:
            panel.text(0.03, max(0.01, y), label, ha='left', va='top', fontsize=fontsize,
                       color=color, fontweight=weight, transform=panel.transAxes)
        # Visualization passively displays supervisor events, never enforces safety.
        self.fig.suptitle('MISSION-ADAPTIVE UAV  /  ROS2 OBSERVATION BRIDGE'
                          if frame.planner and frame.planner.mode == 'ROS_BRIDGE'
                          else 'MISSION-ADAPTIVE UAV  /  STEP 4 + SAFETY-GATED PLANNING',
                          fontsize=14, fontweight='bold', x=.03, ha='left')
        self.fig.subplots_adjust(left=.06, right=.97, top=.86, bottom=.19, wspace=.22)
        self.fig.canvas.draw()

    def save_png(self, frame: VisualizationFrame, destination: Path):
        self.draw(frame)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(destination, dpi=145, facecolor=self.fig.get_facecolor())

    def image(self):
        from PIL import Image
        self.fig.canvas.draw()
        # Image must be copied; Matplotlib reuses the canvas buffer next frame.
        rgba = self.fig.canvas.buffer_rgba()
        return Image.frombytes('RGBA', self.fig.canvas.get_width_height(),
                               bytes(rgba)).convert('RGB')

    def close(self):
        self.plt.close(self.fig)
