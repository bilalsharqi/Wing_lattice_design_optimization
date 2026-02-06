import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import networkx as nx
import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from graph_metrics import find_source_target, compute_boundary_betweenness

# Configure ffmpeg path explicitly
mpl.rcParams["animation.ffmpeg_path"] = (
    r"C:\Users\silber\PycharmProjects\DronesWingsGT\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg.exe"
)

# ---------------- Graph Construction ----------------
def fcc2d_grid(N):
    """Generates a NxN FCC-style grid (square lattice with center points)."""
    points = []
    for ix in range(N + 1):
        for iy in range(N + 1):
            points.append((ix, iy))
    for ix in range(N):
        for iy in range(N):
            points.append((ix + 0.5, iy + 0.5))
    coords = np.array(sorted(set(points)))
    return coords

def fcc2d_graph(N):
    """Create graph (NetworkX object) for the 2D FCC-like grid."""
    coords = fcc2d_grid(N)
    coord_idx = {tuple(pt): i for i, pt in enumerate(coords)}
    G = nx.Graph()
    for i, pt in enumerate(coords):
        G.add_node(i, o=pt)
    # Square lattice edges
    for i, (x, y) in enumerate(coords):
        if (x.is_integer() and y.is_integer()):
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (x+dx, y+dy)
                if nb in coord_idx:
                    j = coord_idx[nb]
                    G.add_edge(i, j)
    # Center-to-corner edges
    for ix in range(N):
        for iy in range(N):
            center = (ix+0.5, iy+0.5)
            if center in coord_idx:
                cidx = coord_idx[center]
                corners = [(ix, iy), (ix+1, iy), (ix, iy+1), (ix+1, iy+1)]
                for c in corners:
                    if c in coord_idx:
                        G.add_edge(cidx, coord_idx[c])
    return G


# ---------------- Boundary Detection ----------------
def get_outer_square_nodes(G, N):
    """
    Return a set of node IDs for the nodes on the *outermost square*.
    This means the nodes whose coordinates are either (x==0 or x==N) OR (y==0 or y==N),
    and, importantly, both must be integer coordinates (not center points).
    """
    outer_nodes = set()
    for n in G.nodes():
        x, y = G.nodes[n]['o']
        # Only consider nodes at the corners or sides (no center points)
        if ((x == 0 or x == N) or (y == 0 or y == N)) and (x.is_integer() and y.is_integer()):
            outer_nodes.add(n)
    return outer_nodes

def get_outer_square_edges(G, N):
    """Return set of edges where BOTH endpoints are on the outermost square."""
    outer_nodes = get_outer_square_nodes(G, N)
    outer_edges = set()
    for u, v in G.edges():
        if u in outer_nodes and v in outer_nodes:
            outer_edges.add(tuple(sorted((u, v))))
    return outer_edges

def is_edge_boundary(G, edge, boundary_nodes):
    """Returns True if either endpoint is a boundary node."""
    u, v = edge
    return u in boundary_nodes or v in boundary_nodes

# ---------------- Updated Edge Removal ----------------
def remove_smallest_edge_betweenness(G, edge_betweenness, protected_edges=None, protected_nodes=None, seed=None):
    """Remove a random edge with smallest betweenness that's not protected and cleanup isolates."""
    if seed is not None:
        random.seed(seed)
    # Format edge-betweenness mapping
    if isinstance(edge_betweenness, dict):
        edge_val_pairs = list(edge_betweenness.items())
    elif isinstance(edge_betweenness, np.ndarray):
        edge_val_pairs = list(zip(G.edges(), edge_betweenness))
    else:
        raise ValueError("edge_betweenness must be a dict or array-like")
    # Exclude protected edges
    if protected_edges:
        edge_val_pairs = [(e,v) for e,v in edge_val_pairs if tuple(sorted(e)) not in protected_edges]
    values = np.array([v for _, v in edge_val_pairs])
    if len(values) == 0:
        return G.copy(), None # No eligible removals
    min_val = np.min(values)
    min_edges = [e for (e, v) in edge_val_pairs if v == min_val]
    edge_to_remove = random.choice(min_edges)
    G2 = G.copy()
    G2.remove_edge(*edge_to_remove)
    protected_nodes = set(protected_nodes or [])
    isolates = [n for n in G2.nodes if G2.degree[n] == 0 and n not in protected_nodes]
    G2.remove_nodes_from(isolates)
    return G2, edge_to_remove

# ---------------- Modified Simulation ----------------
def run_removal_simulation(G, steps, N, percentile=0.01):
    graphs, betweennesses, sources, targets = [], [], [], []
    edge_removed_at = {tuple(sorted(e)): None for e in G.edges()}
    protected_nodes = get_outer_square_nodes(G, N)
    protected_edges = get_outer_square_edges(G, N)
    # protected_edges = set([tuple(sorted(e)) for e in G.edges() if is_edge_boundary(G, e, boundary_nodes)])
    step = 0
    while step <= steps:
        src, tgt = find_source_target(G, percentile=percentile)
        eb = compute_boundary_betweenness(G, src, tgt)
        graphs.append(G.copy())
        betweennesses.append(eb)
        sources.append(src)
        targets.append(tgt)
        # # If graph would disconnect by removing an edge, stop:
        # if not nx.is_connected(G):
        #     print(f"Disconnected at step {step}")
        #     break
        # Remove, but skip protected edges
        if step < steps and len(eb) > 0:
            G_next, removed_edge = remove_smallest_edge_betweenness(
                G, eb, protected_edges=protected_edges, protected_nodes=set(src)|set(tgt)
            )
            if removed_edge is None:
                print("No eligible edge to remove")
                break
            edge_removed_at[tuple(sorted(removed_edge))] = step + 1
            G = G_next
        else:
            break
        step += 1
    return graphs, betweennesses, sources, targets, edge_removed_at

# ---------------- Plotting Helpers ----------------
def plot_betweenness(G, edge_betweenness, sources, targets, ax=None, cmap="plasma", show_colorbar=True, norm=None):
    """Plot graph colored by edge betweenness values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if isinstance(edge_betweenness, dict):
        edge_val_pairs = list(edge_betweenness.items())
    elif isinstance(edge_betweenness, (np.ndarray, list, tuple)):
        edge_val_pairs = list(zip(G.edges(), edge_betweenness))
    else:
        raise ValueError("edge_betweenness must be dict or array-like")

    values = np.array([v for _, v in edge_val_pairs])
    if norm is None:
        norm = mcolors.Normalize(vmin=values.min() if len(values)>0 else 0,
                                 vmax=values.max() if len(values)>0 else 1)

    cmap_obj = plt.get_cmap(cmap)
    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    for (u, v), val in edge_val_pairs:
        x1, y1 = G.nodes[u]['o']
        x2, y2 = G.nodes[v]['o']
        ax.plot([x1, x2], [y1, y2], color=cmap_obj(norm(val)), linewidth=3, zorder=1)

    xs, ys, colors = [], [], []
    for n in G.nodes():
        if n in sources:
            colors.append("red")
        elif n in targets:
            colors.append("blue")
        else:
            colors.append("black")
        x, y = G.nodes[n]['o']
        xs.append(x); ys.append(y)
    ax.scatter(xs, ys, c=colors, s=36, edgecolors="white", linewidths=2, zorder=10)

    ax.set_aspect("equal")
    ax.axis("off")
    if show_colorbar:
        plt.colorbar(smap, ax=ax, label="Edge betweenness")
    return smap

def plot_removal_steps(G, edge_removed_at, cmap="viridis"):
    """Plot initial grid colored by step of removal."""
    removed_steps = [s for s in edge_removed_at.values() if s is not None]
    vmin, vmax = (min(removed_steps), max(removed_steps)) if removed_steps else (0, 1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(6, 6))
    for (u, v) in G.edges():
        x1, y1 = G.nodes[u]['o']
        x2, y2 = G.nodes[v]['o']
        step = edge_removed_at[tuple(sorted((u, v)))]
        color = "lightgray" if step is None else cmap_obj(norm(step))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
    xs, ys = zip(*[G.nodes[n]['o'] for n in G.nodes()])
    ax.scatter(xs, ys, s=20, color="black", zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    plt.colorbar(sm, ax=ax, label="Step removed")
    plt.tight_layout()
    plt.show()

# ---------------- Animation ----------------
def make_animation(graphs, betweennesses, sources, targets, norm, interval=50):
    fig, ax = plt.subplots(figsize=(6, 6))
    cbar = None

    def update(frame):
        nonlocal cbar
        ax.clear()
        smap = plot_betweenness(
            graphs[frame], betweennesses[frame], sources[frame], targets[frame],
            ax=ax, cmap="plasma", show_colorbar=False, norm=norm
        )
        ax.set_title(f"Removal Step {frame}/{len(graphs)-1}")
        if cbar is None:
            cbar = fig.colorbar(smap, ax=ax, label="Edge betweenness",
                                fraction=0.046, pad=0.04)
    ani = FuncAnimation(fig, update, frames=len(graphs), interval=interval, repeat=False)
    return ani

# ---------------- Main Usage Example ----------------
if __name__ == "__main__":
    number_unit_cells = 10
    STEPS = 1000
    PERCENTILE = 0.01

    G0 = fcc2d_graph(number_unit_cells)
    graphs, betweennesses, sources, targets, edge_removed_at = run_removal_simulation(
        G0, STEPS, number_unit_cells, PERCENTILE
    )
    # Compute global norm for betweenness values
    all_vals = []
    for eb in betweennesses:
        if isinstance(eb, dict):
            all_vals.extend(eb.values())
        elif isinstance(eb, (list, tuple, np.ndarray)):
            all_vals.extend(np.ravel(eb))
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Final summary plot
    plot_removal_steps(graphs[0], edge_removed_at)

    # Make and save animation
    ani = make_animation(graphs, betweennesses, sources, targets, norm)
    ani.save("fcc_betweenness_removal.mp4", writer="ffmpeg", fps=10, dpi=150)

