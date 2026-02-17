"""
Graded triangular lattice (graded along x), optionally warped,
converted to strut polygons + fixed-thickness boundary frame,
and solved for strut width W to hit target area fraction phi.

Key changes vs your current version:
- Uses a FIXED frame thickness w_frame (not coupled to W).
- Solves ONLY W (inner struts) using exact union area (Shapely unary_union).
- Polygons: struts built/clipped to OUTER domain (preserves your nice junctions),
  plus frame ring polygon (outer \ inner) for perfect corners + robust welding.
- Graph: strut centerlines clipped to INNER box (inside frame) + 4 inner-perimeter frame edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union

from pattern_project.pattern_generation import (
    merge_tol_from_segments,
    build_igraph_from_segments,
)

# ----------------------------
# Domain helpers
# ----------------------------
def make_rectangle(Lx: float, Ly: float):
    return box(0.0, 0.0, float(Lx), float(Ly))

def pad_rectangle(rect, pad_frac: float = 0.2):
    minx, miny, maxx, maxy = rect.bounds
    W = maxx - minx
    H = maxy - miny
    px = pad_frac * W
    py = pad_frac * H
    return box(minx - px, miny - py, maxx + px, maxy + py)

def _clip_segments_to_rect(starts, ends, rect):
    """Clip centerline segments to a shapely rect. Returns (S,E) arrays."""
    S_out, E_out = [], []
    for a, b in zip(starts, ends):
        inter = LineString([tuple(map(float, a)), tuple(map(float, b))]).intersection(rect)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            coords = list(inter.coords)
            if len(coords) >= 2:
                S_out.append(coords[0]); E_out.append(coords[-1])
        elif inter.geom_type == "MultiLineString":
            for ls in inter.geoms:
                coords = list(ls.coords)
                if len(coords) >= 2:
                    S_out.append(coords[0]); E_out.append(coords[-1])
    if not S_out:
        return np.zeros((0, 2), float), np.zeros((0, 2), float)
    return np.asarray(S_out, float), np.asarray(E_out, float)

def _bar(p0, p1, w, dom=None):
    g = LineString([p0, p1]).buffer(w / 2, cap_style=2, join_style=2)
    return g if dom is None else g.intersection(dom)

# ----------------------------
# Grading s(x)
# ----------------------------
def s_linear_x(x, x_min, x_max, s_min, s_max):
    t = (x - x_min) / (x_max - x_min + 1e-15)
    t = np.clip(t, 0.0, 1.0)
    return s_min + (s_max - s_min) * t

def s_power_x(x, x_min, x_max, s_min, s_max, p=3.0, side="right"):
    t = (x - x_min) / (x_max - x_min + 1e-15)
    t = np.clip(t, 0.0, 1.0)
    if side == "right":
        g = t**p
    elif side == "left":
        g = 1.0 - (1.0 - t) ** p
    else:
        raise ValueError("side must be 'left' or 'right'")
    return s_min + (s_max - s_min) * g

def grading_value(x, x_min, x_max, s_min, s_max, mode="power", p=3.0, side="right"):
    if mode == "linear":
        return s_linear_x(x, x_min, x_max, s_min, s_max)
    if mode == "power":
        return s_power_x(x, x_min, x_max, s_min, s_max, p=p, side=side)
    raise ValueError("mode must be 'linear' or 'power'")

# ----------------------------
# Lattice construction
# ----------------------------
def build_graded_tri_lattice_graph(
    rect,
    s_min, s_max,
    a0=1.0,
    grading_mode="power",
    p=3.0,
    side="right",
):
    minx, miny, maxx, maxy = rect.bounds
    sqrt3 = math.sqrt(3.0)

    nodes, cols_y, cols_idx = [], [], []

    x = minx
    col = 0
    while x <= maxx + 1e-12:
        s_col = float(grading_value(x, minx, maxx, s_min, s_max, mode=grading_mode, p=p, side=side))
        dy = a0 * s_col
        dx = (sqrt3 / 2.0) * dy

        y0 = miny + (0.5 * dy if (col % 2 == 1) else 0.0)
        ys = np.arange(y0, maxy + dy, dy)

        col_inds, col_ys = [], []
        for y in ys:
            if rect.contains(Point(x, y)):
                col_inds.append(len(nodes))
                col_ys.append(y)
                nodes.append((x, y))

        if len(col_inds) >= 2:
            cols_idx.append(np.asarray(col_inds, dtype=int))
            cols_y.append(np.asarray(col_ys, dtype=float))

        x += dx
        col += 1

    nodes = np.asarray(nodes, dtype=float)
    edges = set()

    for idx in cols_idx:
        for k in range(len(idx) - 1):
            i, j = int(idx[k]), int(idx[k + 1])
            edges.add((min(i, j), max(i, j)))

    for c in range(len(cols_idx) - 1):
        y_this, idx_this = cols_y[c], cols_idx[c]
        y_next, idx_next = cols_y[c + 1], cols_idx[c + 1]

        dy_med = float(np.median(np.diff(y_this))) if len(y_this) >= 2 else 0.0
        for y, i in zip(y_this, idx_this):
            i = int(i)
            for target in (y - 0.5 * dy_med, y + 0.5 * dy_med):
                jpos = np.searchsorted(y_next, target)
                cands = []
                if 0 <= jpos < len(y_next): cands.append(jpos)
                if 0 <= jpos - 1 < len(y_next): cands.append(jpos - 1)
                if not cands:
                    continue
                jj = min(cands, key=lambda k: abs(y_next[k] - target))
                j = int(idx_next[jj])
                edges.add((min(i, j), max(i, j)))

    return nodes, sorted(edges)

def edges_to_lines(clip_rect, nodes, edges):
    lines = []
    for i, j in edges:
        p0 = (float(nodes[i, 0]), float(nodes[i, 1]))
        p1 = (float(nodes[j, 0]), float(nodes[j, 1]))
        inter = LineString([p0, p1]).intersection(clip_rect)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            if inter.length > 1e-12:
                lines.append(inter)
        elif inter.geom_type == "MultiLineString":
            for g in inter.geoms:
                if g.length > 1e-12:
                    lines.append(g)
    return lines

# ----------------------------
# Public API
# ----------------------------
@dataclass(frozen=True)
class WarpedLatticeParams:
    Lx: float = 4.3
    Ly: float = 1.0
    pad_frac: float = 0.5

    s_min: float = 0.05
    s_max: float = 0.30
    a0: float = 1.0
    grading_mode: str = "power"
    p: float = 3.0
    side: str = "right"

    phi_target: float = 0.15
    tol: float = 1e-5
    max_iter: int = 30

    add_frame: bool = True
    w_frame: float = 0.02  # FIXED frame thickness


def generate_warped_lattice_polys(params: WarpedLatticeParams):
    """
    Returns:
      polys_final : list[Polygon]
      meta        : dict
      G           : igraph.Graph (x,y + edge length, thickness)
    """
    Lx, Ly = float(params.Lx), float(params.Ly)
    phi_target = float(params.phi_target)
    tol = float(params.tol)
    max_iter = int(params.max_iter)

    outer = make_rectangle(Lx, Ly)
    A_dom = Lx * Ly

    # 1) graded lattice (on padded domain), then centerlines clipped to outer
    dom_pad = pad_rectangle(outer, pad_frac=float(params.pad_frac))
    nodes, edges = build_graded_tri_lattice_graph(
        dom_pad,
        s_min=float(params.s_min),
        s_max=float(params.s_max),
        a0=float(params.a0),
        grading_mode=str(params.grading_mode),
        p=float(params.p),
        side=str(params.side),
    )

    centerlines = edges_to_lines(outer, nodes, edges)
    starts = np.asarray([ln.coords[0] for ln in centerlines], dtype=float)
    ends   = np.asarray([ln.coords[-1] for ln in centerlines], dtype=float)

    # 2) fixed frame ring polygon
    add_frame = bool(params.add_frame)
    w_frame = float(params.w_frame) if add_frame else 0.0

    frame_poly = None
    inner = None
    phi_frame = 0.0
    if add_frame and w_frame > 0:
        if (Lx - 2 * w_frame) <= 0 or (Ly - 2 * w_frame) <= 0:
            frame_poly = outer
            phi_frame = 1.0
        else:
            inner = box(w_frame, w_frame, Lx - w_frame, Ly - w_frame)
            frame_poly = outer.difference(inner)
            phi_frame = float(frame_poly.area) / A_dom

    # 3) exact-phi solve for W (struts only)
    def phi_and_polys(W):
        W = float(W)
        polys = []
        if W > 0 and starts.shape[0] > 0:
            for a, b in zip(starts, ends):
                p = _bar(tuple(a), tuple(b), W, outer)  # OUTER for nice junctions + welding
                if p is not None and not p.is_empty:
                    polys.append(p)
        if frame_poly is not None:
            polys.append(frame_poly)
        if not polys:
            return 0.0, []
        u = unary_union(polys)
        return float(u.area) / A_dom, polys

    if phi_target <= phi_frame + tol:
        W_star = 0.0
        phi_star = phi_frame
        polys_final = [frame_poly] if frame_poly is not None else []
    else:
        lo = 0.0
        hi = max(1e-6, 0.01 * min(Lx, Ly))
        phi_hi, polys_hi = phi_and_polys(hi)
        hard_cap = 0.95 * min(Lx, Ly)

        while phi_hi < phi_target and hi < hard_cap:
            hi *= 2.0
            phi_hi, polys_hi = phi_and_polys(hi)

        if phi_hi < phi_target:
            W_star, phi_star, polys_final = hi, phi_hi, polys_hi
        else:
            W_star = phi_star = None
            polys_final = None
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                phi_mid, polys_mid = phi_and_polys(mid)
                W_star, phi_star, polys_final = mid, phi_mid, polys_mid
                if abs(phi_mid - phi_target) <= tol:
                    break
                if phi_mid < phi_target:
                    lo = mid
                else:
                    hi = mid

    # 4) graph: strut centerlines clipped to INNER box (inside frame) + inner perimeter frame edges
    if inner is not None:
        Sg, Eg = _clip_segments_to_rect(starts, ends, inner)
    else:
        Sg, Eg = starts, ends

    th_g = np.full(Sg.shape[0], float(W_star), dtype=float)

    frame_edges_added = 0
    if inner is not None:
        xmin, ymin, xmax, ymax = w_frame, w_frame, Lx - w_frame, Ly - w_frame
        Sf = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], float)
        Ef = np.array([[xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]], float)
        Sg = np.vstack([Sg, Sf])
        Eg = np.vstack([Eg, Ef])
        th_g = np.concatenate([th_g, np.full(4, float(w_frame), float)])
        frame_edges_added = 4

    merge_tol = merge_tol_from_segments(Sg, Eg, Lx, Ly)
    G = build_igraph_from_segments(Sg, Eg, th_g, merge_tol=merge_tol)

    meta = {
        "Lx": float(Lx), "Ly": float(Ly),
        "mode": str(params.grading_mode),
        "s_min": float(params.s_min),
        "s_max": float(params.s_max),
        "a0": float(params.a0),
        "p": float(params.p),
        "side": str(params.side),
        "W": float(W_star),
        "w_frame": float(w_frame),
        "phi_target": float(phi_target),
        "phi_achieved": float(phi_star),
        "phi_frame": float(phi_frame),
        "add_frame": bool(add_frame),
        "frame_mode": "ring_outer_minus_inner",
        "frame_graph_edges_added": int(frame_edges_added),
        "polys_out": int(len(polys_final)) if polys_final is not None else 0,
        "graph_n": int(G.vcount()),
        "graph_m": int(G.ecount()),
        "merge_tol": float(merge_tol),
        "n_centerlines": int(starts.shape[0]),
        "n_graph_struts": int(Sg.shape[0] - frame_edges_added),
    }
    return polys_final, meta, G
