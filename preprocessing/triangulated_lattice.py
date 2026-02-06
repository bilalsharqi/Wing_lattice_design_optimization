from network_export_to_fabrication import network_to_dxf, render_dxf, extrude_dxf_to_stl, extrude_dxf_to_scad
from graph_metrics import find_source_target_igraph, compute_boundary_betweenness_igraph
from network_pattern import generate_triangulated_lattice_borders
from plotting import plot_polygons, plot_binary_image, plot_graph_with_betweenness_igraph
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from graph_metrics import compute_graph_metrics, export_global_metrics
from plotting import plot_polygons_with_metric_overlay

# ==== PARAMETERS ====
DIVISIONS = 5
SQUARE_SIDE = 1.0
LINE_WIDTH = 0.03
TOL = 1e-6
PERCENTILE = 5
SIGMA_NETWORK = 1

DXF_SIZE_X = 100
DXF_SIZE_Y = 100
Z_HEIGHT = 3
MERGE_RADIUS = 5
DXF_OUTPUT = 'triangulated_network.DXF'
OPENSCAD_PATH = r"C:\Program Files\OpenSCAD\openscad.exe"

def plot_polygons_temp(polygons, side_square=1.0):
    """
    Plots a list of Shapely polygons using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for poly in polygons:
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.7, linewidth=0, color='lightblue')
        else:
            # Handle MultiPolygon or LineString if ever included
            try:
                for p in poly.geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, alpha=0.7, linewidth=0, color='lightblue')
            except Exception:
                continue

    ax.set_aspect('equal', 'box')
    # ax.set_xlim(0, side_square)
    # ax.set_ylim(0, side_square)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Triangulated Lattice Polygons')
    plt.show()

# ===============================================
def generate_lattice(square_side, line_width,number_unit_cells =6, border_width = 0.06):
    """Generate traingulated lattice polygons and graph ."""
    polygons,graph = generate_triangulated_lattice_borders(
        nx = DIVISIONS, ny = DIVISIONS, cell_size=1/DIVISIONS, thickness=0.02, border_width=border_width)
    return polygons, graph

dxf_size_x=DXF_SIZE_X
dxf_size_y=DXF_SIZE_Y
metric_names = {
    'degree': 'Degree',
    'local_clustering': 'Local Clustering Coefficient',
    'betweenness': 'Betweenness Centrality',
    'closeness': 'Closeness Centrality'
    # Add more as needed if your local_metrics dict has more keys
}

def plot_graph_nodes_with_labels(g):
    import numpy as np
    """
    Plot only graph nodes with their graph ID (index) next to each node.

    Args:
        g: igraph.Graph
        source_nodes: list of vertex indices to color red (optional)
        target_nodes: list of vertex indices to color blue (optional)
        ax: matplotlib axis, optional
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    xs = np.array(g.vs['x'])
    ys = np.array(g.vs['y'])

    # Assign colors per node category
    node_colors = []
    for i in range(g.vcount()):
        node_colors.append("black")

    # Plot nodes
    ax.scatter(xs, ys, c=node_colors, s=28, zorder=10,
               edgecolors="white", linewidths=0.8)

    # Plot labels (node IDs)
    for i in range(g.vcount()):
        ax.text(xs[i] + 0.02, ys[i] + 0.02, str(i),
                fontsize=8, color="darkgreen")

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    plt.tight_layout()
    return ax

polys, G = generate_lattice(SQUARE_SIDE, LINE_WIDTH, DIVISIONS, border_width=0.06)

# plot_polygons(polys, ax=None, color='cornflowerblue', edgecolor='k', alpha=0.7)
global_metrics, local_metrics = compute_graph_metrics(G,weighted=True)
# plot_graph_nodes_with_labels(G)
# plot_graph_metrics(G, weighted=True, plot_local=True, fontsize=13)

g = G.copy() #becase the compute_boundary_betweenes_igraph is unstable
source_nodes, target_nodes = find_source_target_igraph(G,percentile=PERCENTILE)
edge_betweenness = compute_boundary_betweenness_igraph(G, source_nodes, target_nodes)

plot_graph_with_betweenness_igraph(
       g, edge_betweenness, source_nodes, target_nodes, edge_width=0.01, ax=None, cmap='plasma'
  )
plt.show()

# for key in local_metrics:
#     nice_name = metric_names.get(key, key.replace('_', ' ').title())
#     plot_polygons_with_metric_overlay(
#         polys, g, local_metrics, key,
#         cmap='plasma',
#         node_size=50,
#         polygon_color='cornflowerblue',
#         polygon_alpha=0.5,
#         font_size=16,
#         output_png_path=f"{nice_name}.png"
#     )

# export_global_metrics(global_metrics, "global_metrics.csv")
# dxf_fname = f"{dxf_size_x}x{dxf_size_y}_triangulated.dxf"
# network_to_dxf(
#     polys,
#     x_size=dxf_size_x, y_size=dxf_size_y,
#     output_path=dxf_fname)
# render_dxf(dxf_fname)
