import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, GeometryCollection
from matplotlib.collections import PolyCollection

def plot_lattice_graph_true_width(
    g,
    *,
    x_attr="x",
    y_attr="y",
    thickness_attr="thickness",          # nominal/bulk (W)
    thickness_eff_attr="thickness_eff",  # optional: polygon-derived / boundary-truncated
    use_effective=False,
    invert_y=False,
    frame_shape=None,
    node_size=8,
    edge_color="black",
    node_color="red",
    edge_alpha=1.0,
):
    """
    Plot an igraph graph by drawing each edge as a rectangle with *true* thickness.

    Intended for your square-lattice output:
      - vertices: x, y
      - edges: thickness (bulk), optional thickness_eff (boundary may differ), optional length

    Parameters
    ----------
    sample : str
        Title.
    g : igraph.Graph
        Graph.
    x_attr, y_attr : str
        Vertex coordinate attributes.
    thickness_attr : str
        Edge thickness attribute (nominal).
    thickness_eff_attr : str
        Alternative thickness attribute (effective).
    use_effective : bool
        If True, use thickness_eff_attr when available; else use thickness_attr.
    invert_y : bool
        If True, reverse y-axis (image convention).
    frame_shape : (H,W) or None
        If provided, set axis limits to (0..W, 0..H) (and invert_y if requested).
    """
    # --- validate attrs ---
    if x_attr not in g.vs.attributes() or y_attr not in g.vs.attributes():
        raise ValueError(f"Graph must have vertex attrs '{x_attr}', '{y_attr}'.")

    use_eff = bool(use_effective and thickness_eff_attr in g.es.attributes())
    w_attr = thickness_eff_attr if use_eff else thickness_attr
    if w_attr not in g.es.attributes():
        raise ValueError(f"Graph must have edge attr '{w_attr}' (and/or '{thickness_attr}').")

    pts = np.column_stack([g.vs[x_attr], g.vs[y_attr]]).astype(float)
    widths = np.asarray(g.es[w_attr], dtype=float)

    fig, ax = plt.subplots()

    # --- draw edges as rectangles ---
    for i, e in enumerate(g.es):
        w = float(widths[i])
        if not np.isfinite(w) or w <= 0:
            continue

        u, v = e.source, e.target
        x0, y0 = pts[u]
        x1, y1 = pts[v]

        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L <= 1e-12:
            continue

        nx, ny = -dy / L, dx / L  # unit normal
        hw = 0.5 * w

        poly = np.array([
            [x0 + nx * hw, y0 + ny * hw],
            [x0 - nx * hw, y0 - ny * hw],
            [x1 - nx * hw, y1 - ny * hw],
            [x1 + nx * hw, y1 + ny * hw],
        ])
        ax.fill(poly[:, 0], poly[:, 1], color=edge_color, alpha=edge_alpha, linewidth=0, zorder=1)
        # ax.fill(poly[:, 0], poly[:, 1], alpha=edge_alpha, linewidth=0, zorder=1) ##debugging: each edge in its own color


    # --- nodes ---
    ax.scatter(pts[:, 0], pts[:, 1], color=node_color, s=node_size, zorder=2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_attr)
    ax.set_ylabel(y_attr)

    # --- limits / y inversion ---
    if frame_shape is not None:
        H, W = frame_shape
        ax.set_xlim(0, W)
        if invert_y:
            ax.set_ylim(H, 0)
        else:
            ax.set_ylim(0, H)
    else:
        if invert_y:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(max(y0, y1), min(y0, y1))

    # plt.show()



def plot_points(rect, pts, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, y = rect.exterior.xy
    ax.plot(x, y, linewidth=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=6)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_lines(rect, lines, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, y = rect.exterior.xy
    ax.plot(x, y, linewidth=2)
    for ln in lines:
        xs, ys = ln.xy
        ax.plot(xs, ys, linewidth=0.8)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_material(
    obj, Lx, Ly, title=None,
    ax=None, figsize=None,
    facecolor="C0", edgecolor="none", lw=0.2,
    show_rect=False, rect_lw=1.0,
    show_axes=False,
    holes="white",   # "white" or "ignore"
):
    """
    Plot material polygons.

    obj: either
      - list of shapely Polygons (polys)
      - a shapely Polygon/MultiPolygon/GeometryCollection (poly)

    holes:
      - "white": render holes as white patches (requires union geometry)
      - "ignore": fastest; plots only exteriors (holes may appear filled)
    """

    if ax is None:
        if figsize is None:
            aspect = float(Lx) / float(Ly)
            figsize = (aspect * 3, 3)
        fig, ax = plt.subplots(figsize=figsize)

    # --- normalize input ---
    if isinstance(obj, (list, tuple)):
        polys = [p for p in obj if p is not None and not p.is_empty]
        geom = unary_union(polys) if polys else Polygon()
    else:
        geom = obj
        polys = None

    # --- optional domain rectangle ---
    if show_rect:
        ax.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], color="k", linewidth=rect_lw)

    # --- plotting paths ---
    if holes == "white":
        # hole-aware: iterate union geometry, fill exterior, then fill holes white
        def fill_one(p: Polygon):
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=0.9)
            for hole in p.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, facecolor="white", edgecolor="none")

        if geom and (not geom.is_empty):
            if isinstance(geom, Polygon):
                fill_one(geom)
            elif isinstance(geom, (MultiPolygon, GeometryCollection)):
                for g in geom.geoms:
                    if isinstance(g, Polygon) and (not g.is_empty):
                        fill_one(g)

    else:
        # fastest: PolyCollection of exteriors only (holes ignored)
        verts = []

        def add_geom(g):
            if g.is_empty:
                return
            if isinstance(g, Polygon):
                x, y = g.exterior.coords.xy
                verts.append(np.column_stack([x, y]))
            elif isinstance(g, (MultiPolygon, GeometryCollection)):
                for gg in g.geoms:
                    add_geom(gg)

        if polys is not None:
            for p in polys:
                add_geom(p)
        else:
            add_geom(geom)

        if verts:
            pc = PolyCollection(
                verts, closed=True,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=lw
            )
            ax.add_collection(pc)

    # --- axes formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)

    if title is not None:
        ax.set_title(title)

    if not show_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()
    return ax
