from network_export_to_fabrication import network_to_dxf, render_dxf, extrude_dxf_to_stl, extrude_dxf_to_scad
from graph_metrics import compute_boundary_betweenness_igraph
from network_pattern import generate_triangulated_lattice_borders
from plotting import plot_polygons, plot_binary_image, plot_graph_with_betweenness_igraph
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from graph_metrics import compute_graph_metrics, export_global_metrics, find_source_target_igraph
from load_polygons_from_dxf import dxf_hatch_to_polygon, plot_dxf_lwpolylines, extract_polygons_from_dxf
from graph_construction import render_network_cv
import cv2
import numpy as np
from graph_construction import extract_graph_from_image, calculate_graph_weights_transport
from plotting import plot_graph_on_image, plot_edges_with_width
import igraph as ig
from shapely.ops import unary_union
# ==== PARAMETERS ====
DIVISIONS = 5
SQUARE_SIDE = 1.0
LINE_WIDTH = 0.03
TOL = 1e-6
PERCENTILE = 15
SIGMA_NETWORK = 1

DXF_SIZE_X = 80
DXF_SIZE_Y = 80
Z_HEIGHT = 3
MERGE_RADIUS = 20
DXF_OUTPUT = 'delaunay_network.DXF'
OPENSCAD_PATH = r"C:\Program Files\OpenSCAD\openscad.exe"

def invert_voids(rectangle: Polygon, void_polygons):
    """
    rectangle: Shapely Polygon representing the full domain.
    void_polygons: list of Shapely Polygons representing holes (voids).

    Returns:
        A list of Shapely Polygons representing the remaining solid regions.
    """
    # No voids → the solid is the rectangle
    if not void_polygons:
        return [rectangle]

    # Merge voids so geometry difference is clean
    merged_voids = unary_union(void_polygons)

    # Subtract voids from the rectangle
    solid = rectangle.difference(merged_voids)

    # The result may be a Polygon or MultiPolygon
    if solid.geom_type == "Polygon":
        return [solid]
    elif solid.geom_type == "MultiPolygon":
        return list(solid.geoms)
    else:
        raise ValueError(f"Unexpected geometry type: {solid.geom_type}")

def plot_inverted_polygons(polygons):
    network_lines = [poly.exterior for poly in polygons]
    for ring in network_lines:
        x, y = ring.xy
        plt.plot(x, y, linewidth=1)


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
    ax.set_title('Delaunay Network Polygons')
    # plt.show()

def render_solid_from_polygons(polygons, img_size=1024):
    """
    Render a binary image of the solid region defined by the first polygon (boundary)
    minus all inner polygons (holes).

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        First polygon is the boundary, subsequent ones are holes.
    img_size : int
        Output image size (square).

    Returns
    -------
    img : np.ndarray (uint8)
        Binary image (255 = solid, 0 = background)
    """
    if len(polygons) == 0:
        raise ValueError("No polygons provided")

    # --- Collect coordinates to normalize ---
    all_coords = np.vstack([np.array(p.exterior.coords) for p in polygons])
    min_xy = all_coords.min(axis=0)
    max_xy = all_coords.max(axis=0)
    range_xy = max_xy - min_xy

    # --- Initialize image ---
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # --- Normalize helper ---
    def normalize_coords(verts):
        verts = (verts - min_xy) / (range_xy + 1e-12)
        contour = np.stack([
            (verts[:, 0] * (img_size - 1)).astype(np.int32),
            ((1 - verts[:, 1]) * (img_size - 1)).astype(np.int32)
        ], axis=1)
        return contour

    # --- Draw the outer boundary as white ---
    boundary_poly = polygons[0]
    boundary_contour = normalize_coords(np.array(boundary_poly.exterior.coords))
    cv2.fillPoly(img, [boundary_contour], color=255)

    # --- Subtract holes (draw as black) ---
    for poly in polygons[1:]:
        hole_contour = normalize_coords(np.array(poly.exterior.coords))
        cv2.fillPoly(img, [hole_contour], color=0)

    return img

def add_nodes_to_graph(graph, new_nodes):
    """
    Add one or more new nodes to a skeleton graph, splitting edges as needed.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph (typically from sknw.build_sknw).
    new_nodes : list of tuple
        List of (x, y) coordinates of new nodes to insert.

    Returns
    -------
    graph : networkx.Graph
        Updated graph with the new nodes and split edges.
    """

    def find_closest_edge(graph, point):
        px, py = point
        min_dist = float("inf")
        closest_edge = None
        closest_point_on_edge = None

        for u, v, data in graph.edges(data=True):
            pts = np.array(data.get("pts"))
            if pts is None or len(pts) < 2:
                continue

            # Compute closest point on polyline
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                vx, vy = x2 - x1, y2 - y1
                wx, wy = px - x1, py - y1
                t = max(0, min(1, (vx * wx + vy * wy) / (vx**2 + vy**2 + 1e-9)))
                proj = np.array([x1 + t * vx, y1 + t * vy])
                dist = np.linalg.norm(proj - np.array([px, py]))
                if dist < min_dist:
                    min_dist = dist
                    closest_edge = (u, v)
                    closest_point_on_edge = proj

        return closest_edge, closest_point_on_edge

    for point in new_nodes:
        edge, closest_point = find_closest_edge(graph, point)
        if edge is None or closest_point is None:
            continue

        u, v = edge
        pts = np.array(graph[u][v]["pts"])
        dists = np.linalg.norm(pts - closest_point, axis=1)
        split_idx = np.argmin(dists)

        # Add the new node
        new_node = max(graph.nodes) + 1
        graph.add_node(new_node, o=closest_point)

        # Create new edge paths
        pts1 = pts[:split_idx + 1]
        pts2 = pts[split_idx:]

        # Remove old edge and add two new ones
        graph.remove_edge(u, v)
        graph.add_edge(u, new_node, pts=pts1)
        graph.add_edge(new_node, v, pts=pts2)

    return graph

def nx_to_igraph(nx_graph):
    """
    Convert a NetworkX graph (e.g., from sknw) to an igraph.Graph.
    Keeps node coordinates and edge weights.

    Parameters
    ----------
    nx_graph : networkx.Graph
        The input NetworkX graph, typically from skeleton extraction.

    Returns
    -------
    ig_graph : igraph.Graph
        The converted igraph.Graph with equivalent structure.
    """

    # Create igraph with the same number of nodes
    ig_graph = ig.Graph()
    ig_graph.add_vertices(nx_graph.number_of_nodes())

    # Add edges
    ig_graph.add_edges(list(nx_graph.edges()))

    # Copy node attributes (coordinates, etc.)
    node_attrs = {}
    for node, data in nx_graph.nodes(data=True):
        for key, val in data.items():
            node_attrs.setdefault(key, [None] * nx_graph.number_of_nodes())
            node_attrs[key][node] = val

    for key, vals in node_attrs.items():
        ig_graph.vs[key] = vals

    # Copy edge attributes (weights, pts, etc.)
    edge_attrs = {}
    for i, (u, v, data) in enumerate(nx_graph.edges(data=True)):
        for key, val in data.items():
            edge_attrs.setdefault(key, [None] * nx_graph.number_of_edges())
            edge_attrs[key][i] = val

    for key, vals in edge_attrs.items():
        ig_graph.es[key] = vals

    # Add "weight" if not already defined
    if "weight" not in ig_graph.es.attribute_names():
        ig_graph.es["weight"] = 0.0

    coords = np.array(ig_graph.vs["o"])  # shape: (num_nodes, 2)
    ig_graph.vs["y"] = coords[:, 0].tolist() # rotated for EBC_LB
    ig_graph.vs["x"] = coords[:, 1].tolist()

    return ig_graph

def polygons_to_image_coords(polygons, img_size=1024, input_bounds=[-40, 40, -40, 40]):
    """
    Map polygons from physical coordinates to image coordinates without flipping y-axis.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Input polygons in physical coordinates.
    img_size : int
        Output image size (square).
    input_bounds : list
        [xmin, xmax, ymin, ymax] of the input coordinate system.

    Returns
    -------
    img_polygons : list of shapely.geometry.Polygon
        Polygons in image coordinates (0 to img_size-1).
    """
    xmin, xmax, ymin, ymax = input_bounds
    scale_x = (img_size - 1) / (xmax - xmin)
    scale_y = (img_size - 1) / (ymax - ymin)

    img_polygons = []

    for poly in polygons:
        coords = np.array(poly.exterior.coords)
        x_img = (coords[:, 0] - xmin) * scale_x
        y_img = (coords[:, 1] - ymin) * scale_y  # no flip
        img_poly = Polygon(np.column_stack([x_img, y_img]))
        img_polygons.append(img_poly)

    return img_polygons

def plot_polygons_with_metric_overlay(
    polys,
    g,
    local_metrics,
    metric_name,
    cmap='viridis',
    node_size=50,
    edge_width=2,
    edge_alpha=0.6,
    font_size=16,
    polygon_color='cornflowerblue',
    polygon_edgecolor='k',
    polygon_alpha=0.7,
    output_png_path=None,
    img_size=1024
):

    # Plot polygons as background
    # plot_polygons(polys)
    ax = plt.gca()

    # Metrics
    vals = np.array(local_metrics[metric_name])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    # Rotate/flip graph coordinates to match OpenCV image orientation
    xs_phys = np.array(g.vs["x"])
    ys_phys = np.array(g.vs["y"])
    xs_img = ys_phys                  # old y -> new x
    ys_img = img_size - xs_phys       # old x flipped vertically

    # Overlay nodes
    nodes = ax.scatter(xs_img, ys_img, c=vals, cmap=cmap, s=node_size,
                       vmin=vmin, vmax=vmax, edgecolors='w', linewidths=0.8, zorder=10)

    # Draw edges
    for e in g.es:
        u, v = e.tuple
        x0, y0 = g.vs[u]["y"], img_size - g.vs[u]["x"]
        x1, y1 = g.vs[v]["y"], img_size - g.vs[v]["x"]
        ax.plot([x0, x1], [y0, y1], color='k', alpha=edge_alpha,
                linewidth=edge_width, zorder=3)

    # Title & colorbar
    avg_val = np.nanmean(vals)
    ax.set_title(f"{metric_name.replace('_',' ').title()}\nAverage: {avg_val:.3f}",
                 fontsize=font_size)
    ax.set_aspect('equal')
    ax.axis('off')

    cbar = plt.colorbar(nodes, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(metric_name.replace('_',' ').title(), fontsize=font_size-3)
    cbar.ax.tick_params(labelsize=font_size-5)

    plt.tight_layout()
    if output_png_path:
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    # plt.show()
    return ax

def _map_coords(coords, xmin, ymin, scale_x, scale_y, clip_to=None):
    coords = np.asarray(coords)
    x_img = (coords[:, 0] - xmin) * scale_x
    y_img = (coords[:, 1] - ymin) * scale_y
    mapped = np.column_stack([x_img, y_img])
    if clip_to is not None:
        mapped[:, 0] = np.clip(mapped[:, 0], 0, clip_to)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, clip_to)
    return [tuple(pt) for pt in mapped]

def inverted_polygons_to_image_coords(polygons, img_size=1024, input_bounds=[-40, 40, -40, 40], clip=True):
    """
    Map polygons from physical coordinates to image coordinates while preserving holes.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon or MultiPolygon (or a single Polygon)
        Input polygons in physical coordinates.
    img_size : int
        Output image size (square).
    input_bounds : list
        [xmin, xmax, ymin, ymax] of the input coordinate system.
    clip : bool
        Clip coordinates to [0, img_size-1] to avoid tiny out-of-bounds coords.

    Returns
    -------
    img_polygons : list of shapely.geometry.Polygon
        Polygons in image coordinates (0 .. img_size-1) with interiors preserved.
    """
    xmin, xmax, ymin, ymax = input_bounds
    if xmax == xmin or ymax == ymin:
        raise ValueError("input_bounds must have non-zero extent")
    scale_x = (img_size - 1) / (xmax - xmin)
    scale_y = (img_size - 1) / (ymax - ymin)
    clip_to = img_size - 1 if clip else None

    # normalize single polygon input
    if isinstance(polygons, (Polygon, MultiPolygon)):
        poly_iter = [polygons]
    else:
        poly_iter = polygons

    out_polys = []
    for item in poly_iter:
        if item is None or item.is_empty:
            continue

        if item.geom_type == "Polygon":
            polys_to_process = [item]
        elif item.geom_type == "MultiPolygon":
            polys_to_process = list(item.geoms)
        else:
            # skip non-polygon geometry
            continue

        for poly in polys_to_process:
            # map exterior
            exterior_mapped = _map_coords(poly.exterior.coords, xmin, ymin, scale_x, scale_y, clip_to)

            # map interiors (holes)
            interiors_mapped = []
            for interior in poly.interiors:
                im = _map_coords(interior.coords, xmin, ymin, scale_x, scale_y, clip_to)
                # only keep interiors with at least 3 distinct points
                if len(im) >= 3:
                    interiors_mapped.append(im)

            try:
                newp = Polygon(exterior_mapped, interiors_mapped)
                if not newp.is_valid:
                    newp = newp.buffer(0)  # try to fix tiny geometry issues
                if not newp.is_empty:
                    out_polys.append(newp)
            except Exception:
                # fallback: try create polygon from exterior only (shouldn't usually happen)
                try:
                    fallback = Polygon(exterior_mapped)
                    if fallback.is_valid and not fallback.is_empty:
                        out_polys.append(fallback)
                except Exception:
                    pass

    return out_polys
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
dxf_file_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\delaunay\V1\80x80_delaunay.dxf"
new_nodes = [(10,7),(10,1000), (1011,9)]

polys, lines = extract_polygons_from_dxf(dxf_file_path)
polys_invert = invert_voids(polys[0],polys[1:])
# plot_polygons(polys_invert, show_box = False)
polys = polygons_to_image_coords(polys)
polys_invert = inverted_polygons_to_image_coords(polys_invert)
# plot_polygons(polys_invert)
# plt.show()
binary_image = render_solid_from_polygons(polys)
polys = polys[1:]
plot_binary_image(binary_image)
graph = extract_graph_from_image(binary_image, merge_radius = 30)
graph = add_nodes_to_graph(graph,new_nodes)
graph = calculate_graph_weights_transport(graph,binary_image, inverse_flag=True) #EBC_LB weights is length / width
# # plot_graph_on_image(binary_image, graph)
G = nx_to_igraph(graph)
global_metrics, local_metrics = compute_graph_metrics(G,weighted=True)
g = G.copy() #becase the compute_boundary_betweenes_igraph is unstable
# source_nodes, target_nodes = find_source_target_igraph(G,percentile=PERCENTILE)
# edge_betweenness = compute_boundary_betweenness_igraph(G, source_nodes, target_nodes)
# plt.show()
import csv

# Assuming you already have G as your igraph.Graph

#
# from shapely import affinity
# polys_invert = [affinity.rotate(polygon, angle = 90, origin=(0,0)) for polygon in polys_invert]
# for key in local_metrics:
#     plt.figure()
#     nice_name = metric_names.get(key, key.replace('_', ' ').title())
#     plot_polygons_with_metric_overlay(
#         polys_invert, g, local_metrics, key,
#         cmap='viridis',
#         node_size=50,
#         polygon_color='cornflowerblue',
#         polygon_alpha=0.5,
#         font_size=16,
#         output_png_path=f"{nice_name}.png"
#     )
# # plot_graph_with_betweenness_igraph(
# #        g, edge_betweenness, target_nodes, source_nodes, edge_width=15, ax=None, cmap='plasma'
# #   )
# plt.show()
# # # export_global_metrics(global_metrics, "global_metrics.csv")
