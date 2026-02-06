from network_export_to_fabrication import network_to_dxf, render_dxf, extrude_dxf_to_stl, extrude_dxf_to_scad
from graph_metrics import find_source_target_igraph, compute_boundary_betweenness_igraph
from network_pattern import hex_lattice_bar_polygons
from network_pattern import hex_lattice_centerline_graph_clipped_igraph
from plotting import plot_polygons, plot_binary_image, plot_graph_with_betweenness_igraph
from plotting import plot_polygons_with_graph
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from graph_metrics import compute_graph_metrics, export_global_metrics
from plotting import plot_polygons_with_metric_overlay
from metric_graph_eigenmodes import igraph_to_metric_graph,find_eigenvalues_fast,plot_eigenvalues_with_bounds
from metric_graph_eigenmodes import plot_spectral_gap_bounds, get_mode_vector_fast,plot_mode, precompute_M_template
from metric_graph_eigenmodes import plot_metric_graph_mode, plot_metric_graph_mode_analytic

from graph_metrics import compute_metric_graph_diameter

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
DXF_OUTPUT = 'hexagonal_network.DXF'
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

# polys = hex_lattice_bar_polygons(D0 = 0.02, density = 0.4, domain_L=1.0, crop_eps=1e-6)
polys, G, nodes, edges = hex_lattice_centerline_graph_clipped_igraph(D0=0.02, density = 0.4, domain_L=1.0)
edges_directed, lengths = igraph_to_metric_graph(G,nodes,edges)
E = len(edges_directed)
V = len(nodes)

L = lengths.sum()  # total length
D = compute_metric_graph_diameter(V, edges, lengths)

template = precompute_M_template(V, edges)
ks, sig, eigs  = find_eigenvalues_fast(template, lengths, kmin=2, kmax=6, num=200)
filename = 'hexagonal_lattice_eigenvalues.pkl'
import pickle
# Open the file in binary write mode ('wb') and dump the list
with open(filename, 'wb') as f:
    pickle.dump(eigs, f)

print(f"List saved to {filename}")
print(eigs[:10])
# Plot first few modes
# for k in eigs:
#     print(k)
#     A = get_mode_vector_fast(template, lengths, k)
#     A_undirected = A[:V]  # only take the first V elements
#     # # Now plot
#     # plot_metric_graph_mode(nodes, edges, A_undirected, polys=polys)
#
#     plot_metric_graph_mode_analytic(
#     nodes=nodes,
#     edges=edges,              # undirected edges (length E)
#     A_nodes=A_undirected,     # mode amplitude per node
#     lengths=lengths[:len(edges)], # lengths corresponding to undirected edges
#     k=k,                # first eigenvalue
#     polys=polys,
#     num_points=30
# )

# plot_eigenvalues_with_bounds(eigs, L)
# plot_spectral_gap_bounds(eigs[0], E, V, L, D)
# # plot_polygons_with_graph(polys,nodes,edges)
#
# Plot sigma_min scan with eigenvalues marked
plt.figure(figsize=(8, 4))
plt.plot(ks, sig, 'o-', label=r"$\sigma_{\min}(k)$")
# plt.yscale("log")
for k in eigs:
    plt.axvline(k, color="red", linestyle="--", alpha=0.7)
plt.title("Sigma_min scan + eigenvalues")
plt.xlabel("k")
plt.ylabel(r"$\sigma_{\min}$")
plt.tight_layout()
plt.show()

# plot_polygons(polys, ax=None, color='cornflowerblue', edgecolor='k', alpha=0.7)
# global_metrics, local_metrics = compute_graph_metrics(G,weighted=True)
# # plot_graph_metrics(G, weighted=True, plot_local=True, fontsize=13)
#
# g = G.copy() #becase the compute_boundary_betweenes_igraph is unstable
# source_nodes, target_nodes = find_source_target_igraph(G,percentile=PERCENTILE)
# edge_betweenness = compute_boundary_betweenness_igraph(G, source_nodes, target_nodes)
# plot_graph_with_betweenness_igraph(
#        g, edge_betweenness, source_nodes, target_nodes, edge_width=0.01, ax=None, cmap='plasma'
#   )
#
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
#
plt.show()
#
# export_global_metrics(global_metrics, "global_metrics.csv")
# dxf_fname = f"{dxf_size_x}x{dxf_size_y}_hexagonal.dxf"
# network_to_dxf(
#     polys,
#     x_size=dxf_size_x, y_size=dxf_size_y,
#     output_path=dxf_fname)
# render_dxf(dxf_fname)
