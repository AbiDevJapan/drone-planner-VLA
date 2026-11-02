"""Optional ground-truth-safe visualization of observed occupancy and frontiers."""
from pathlib import Path
from .grid import UNKNOWN, FREE, OCCUPIED


def save_discovered_map(sim, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    snap = sim.snapshot()
    # The figure takes only sim.map.known, never sim.map.terrain.
    values = [[{UNKNOWN: 0, FREE: 1, OCCUPIED: 2}[x] for x in row]
              for row in sim.map.known]
    cmap = ListedColormap(['#d8dce1','#f5f7fa','#263445'])
    norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(values, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    ax.set_title(f'{sim.scenario.name} | Step 2 observed occupancy (map version {sim.map_version})')
    ax.set_xlabel('Map column (m)')
    ax.set_ylabel('Map row (m)')
    frontiers = [sim._from_pos(f.observation_pose) for f in snap.frontiers]
    if frontiers:
        ax.scatter([p[1] for p in frontiers], [p[0] for p in frontiers], c='#d68121',
                   marker='s', s=60, edgecolors='white', linewidths=0.5, label='Frontier viewpoint')
    if sim.path_history:
        ax.plot([p[1] for p in sim.path_history], [p[0] for p in sim.path_history],
                color='#157b9d', linewidth=1.8, alpha=0.85, label='Executed known-free path')
    for entity in snap.entities:
        p = sim._from_pos(entity.position)
        ax.scatter(p[1],p[0],marker='*',s=190,color='#d84f3f',edgecolors='white',label='Discovered target')
    ax.scatter(sim.home[1], sim.home[0], marker='P', s=140, color='#17965b', label='Home')
    ax.scatter(sim.position[1], sim.position[0], marker='o', s=130,
               color='#6443a6', edgecolors='white', label='Drone')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels,handles))
    by_label.update({
        'Unknown': Patch(facecolor='#d8dce1'),
        'Free': Patch(facecolor='#f5f7fa',edgecolor='#a0a5b0'),
        'Occupied': Patch(facecolor='#263445')
    })
    ax.legend(by_label.values(),by_label.keys(),bbox_to_anchor=(1.03,1),loc='upper left',fontsize=8)
    ax.set_xticks(range(0,sim.map.width,2))
    ax.set_yticks(range(0,sim.map.height,2))
    ax.grid(color='#a7b5c2',alpha=0.26)
    fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=155,bbox_inches='tight')
    plt.close(fig)
