import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, GeometryCollection
from matplotlib.collections import PolyCollection

# ----------------------------
# Plot helpers
# ----------------------------
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
    # plt.show()
    return ax
