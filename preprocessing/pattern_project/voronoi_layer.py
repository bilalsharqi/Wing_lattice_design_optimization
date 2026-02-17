import numpy as np
import igraph as ig
from scipy.spatial import Voronoi
from shapely.geometry import box, LineString
from shapely.ops import unary_union


# ----------------------------- geometry helpers -----------------------------

def _bar(p0, p1, w, dom=None):
    g = LineString([p0, p1]).buffer(w / 2, cap_style=2, join_style=2)
    return g if dom is None else g.intersection(dom)


def _clip_segments_to_box(starts, ends, xmin, ymin, xmax, ymax):
    """Clip line segments to [xmin,xmax]x[ymin,ymax]. Returns (S,E) arrays."""
    dom = box(float(xmin), float(ymin), float(xmax), float(ymax))
    S_out, E_out = [], []
    for a, b in zip(starts, ends):
        inter = LineString([tuple(map(float, a)), tuple(map(float, b))]).intersection(dom)
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


def initialize_points_rect(n_points, Lx, Ly, seed=None):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.random(n_points) * float(Lx), rng.random(n_points) * float(Ly)])


def voronoi_ridge_segments(points, Lx, Ly, extend_factor=10.0):
    """
    Voronoi ridge centerlines as segments. Infinite ridges are extended and later clipped.
    Returns: starts, ends, vor
    """
    vor = Voronoi(points)
    center = points.mean(axis=0)
    diag = float(np.hypot(Lx, Ly))

    starts, ends = [], []
    for (pi, pj), v_idx in zip(vor.ridge_points, vor.ridge_vertices):
        v_idx = np.asarray(v_idx, int)

        if np.all(v_idx >= 0):
            a, b = vor.vertices[v_idx[0]], vor.vertices[v_idx[1]]
            starts.append(a); ends.append(b)
            continue

        finite = v_idx[v_idx >= 0]
        if finite.size != 1:
            continue
        v0 = vor.vertices[finite[0]]

        t = points[pj] - points[pi]
        nrm = np.linalg.norm(t)
        if nrm == 0:
            continue
        t /= nrm
        n = np.array([-t[1], t[0]], float)

        mid = 0.5 * (points[pi] + points[pj])
        direction = n if np.dot(mid - center, n) > 0 else -n
        far = v0 + direction * (extend_factor * diag)

        starts.append(v0); ends.append(far)

    return np.asarray(starts, float), np.asarray(ends, float), vor


# ----------------------------- graph helpers -----------------------------

def merge_tol_from_segments(starts, ends, Lx, Ly):
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    diag = float(np.hypot(Lx, Ly))
    if starts.size == 0:
        return max(1e-12, 1e-8 * diag)
    L = np.hypot(ends[:, 0] - starts[:, 0], ends[:, 1] - starts[:, 1])
    L = L[np.isfinite(L) & (L > 0)]
    if L.size == 0:
        return max(1e-12, 1e-8 * diag)
    L1 = float(np.percentile(L, 1.0))
    return max(1e-8 * diag, 1e-3 * L1)


def build_igraph_from_segments(starts, ends, thicknesses, *, merge_tol):
    """
    Vertex attrs: x,y
    Edge attrs: length, thickness
    """
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    th     = np.asarray(thicknesses, float)

    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts/ends must be (M,2) arrays.")
    if th.shape[0] != starts.shape[0]:
        raise ValueError("thicknesses must have length M.")
    q = float(merge_tol)
    if q <= 0:
        raise ValueError("merge_tol must be > 0.")

    grid, xs, ys = {}, [], []
    edges, lengths, ths = [], [], []

    def cell(p):
        return (int(np.floor(p[0] / q)), int(np.floor(p[1] / q)))

    def get_vid(p):
        c = cell(p)
        best, best_d2 = None, q * q
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for vid in grid.get((c[0] + dx, c[1] + dy), []):
                    d2 = (xs[vid] - p[0]) ** 2 + (ys[vid] - p[1]) ** 2
                    if d2 < best_d2:
                        best_d2, best = d2, vid
        if best is not None:
            return best
        vid = len(xs)
        xs.append(float(p[0])); ys.append(float(p[1]))
        grid.setdefault(c, []).append(vid)
        return vid

    for a, b, w in zip(starts, ends, th):
        u = get_vid(a); v = get_vid(b)
        if u == v:
            continue
        L = float(np.hypot(xs[v] - xs[u], ys[v] - ys[u]))
        if L <= 1e-12:
            continue
        edges.append((u, v))
        lengths.append(L)
        ths.append(float(w))

    G = ig.Graph(n=len(xs), edges=edges, directed=False)
    G.vs["x"], G.vs["y"] = xs, ys
    G.es["length"], G.es["thickness"] = lengths, ths
    return G


# ----------------------------- main generator -----------------------------

def generate_random_voronoi_network(
    Lx=16.0, Ly=1.0, n_points=400, phi_target=0.05,
    seed=None, tol=1e-3, max_iter=30, clip=True,
    add_frame=True, w_frame=0.02
):
    """
    Preserves the "old" Voronoi look (no inner clipping for POLYGONS),
    uses a fixed-thickness frame ring (w_frame), solves W for total phi,
    and returns polys, meta, G.

    Polys:
      union( Voronoi bars clipped to outer ) U (outer \ inner(w_frame))

    Graph:
      Voronoi centerlines clipped to inner box (inside frame) + 4 inner-perimeter edges (frame).
    """
    Lx, Ly = float(Lx), float(Ly)
    phi_target = float(phi_target)
    w_frame = float(w_frame)

    outer = box(0.0, 0.0, Lx, Ly)
    A_dom = Lx * Ly

    # Voronoi ridges
    points = initialize_points_rect(n_points, Lx, Ly, seed=seed)
    starts, ends, _vor = voronoi_ridge_segments(points, Lx, Ly)

    # Polygon bars use OUTER-clipped centerlines (keeps your nice junctions)
    if clip:
        S_poly, E_poly = _clip_segments_to_box(starts, ends, 0.0, 0.0, Lx, Ly)
    else:
        S_poly, E_poly = np.asarray(starts, float), np.asarray(ends, float)

    # Frame ring polygon (perfect corners, fully inside)
    frame_poly = None
    phi_frame = 0.0
    inner = None
    if add_frame and w_frame > 0 and (Lx - 2 * w_frame) > 0 and (Ly - 2 * w_frame) > 0:
        inner = box(w_frame, w_frame, Lx - w_frame, Ly - w_frame)
        frame_poly = outer.difference(inner)
        phi_frame = float(frame_poly.area) / A_dom
    elif add_frame and w_frame > 0:
        frame_poly = outer
        phi_frame = 1.0

    if phi_target <= phi_frame + tol:
        polys = [frame_poly] if frame_poly is not None else []
        W = 0.0
        phi_ach = phi_frame
    else:
        def phi_and_polys(W):
            W = float(W)
            polys = []
            if W > 0:
                for a, b in zip(S_poly, E_poly):
                    p = _bar(tuple(a), tuple(b), W, outer)
                    if p is not None and not p.is_empty:
                        polys.append(p)
            if frame_poly is not None:
                polys.append(frame_poly)
            u = unary_union(polys) if polys else None
            return (float(u.area) / A_dom if u is not None else 0.0), polys

        # bracket
        lo, hi = 0.0, max(1e-6, 0.01 * min(Lx, Ly))
        phi_hi, polys_hi = phi_and_polys(hi)
        hard_cap = 0.95 * min(Lx, Ly)
        while phi_hi < phi_target and hi < hard_cap:
            hi *= 2.0
            phi_hi, polys_hi = phi_and_polys(hi)

        if phi_hi < phi_target:
            W, phi_ach, polys = hi, phi_hi, polys_hi
        else:
            W = phi_ach = None
            polys = None
            for _ in range(int(max_iter)):
                mid = 0.5 * (lo + hi)
                phi_mid, polys_mid = phi_and_polys(mid)
                W, phi_ach, polys = mid, phi_mid, polys_mid
                if abs(phi_mid - phi_target) <= tol:
                    break
                (lo, hi) = (mid, hi) if (phi_mid < phi_target) else (lo, mid)

    # ---------------- Graph (inside frame) ----------------
    # Voronoi graph uses INNER-clipped centerlines so edges don't appear in the frame annulus
    frame_edges_added = 0
    if inner is not None:
        xmin, ymin, xmax, ymax = w_frame, w_frame, Lx - w_frame, Ly - w_frame
        S_g, E_g = _clip_segments_to_box(starts, ends, xmin, ymin, xmax, ymax)
    else:
        S_g, E_g = _clip_segments_to_box(starts, ends, 0.0, 0.0, Lx, Ly)

    th_g = np.full(S_g.shape[0], float(W), dtype=float)

    # Add inner-perimeter frame skeleton (4 edges)
    if inner is not None:
        Sf = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], float)
        Ef = np.array([[xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]], float)
        S_g = np.vstack([S_g, Sf])
        E_g = np.vstack([E_g, Ef])
        th_g = np.concatenate([th_g, np.full(4, float(w_frame), float)])
        frame_edges_added = 4

    merge_tol = merge_tol_from_segments(S_g, E_g, Lx, Ly)
    G = build_igraph_from_segments(S_g, E_g, th_g, merge_tol=merge_tol)

    meta = {
        "Lx": Lx, "Ly": Ly,
        "n_points": int(n_points),
        "n_segments_raw": int(np.asarray(starts).shape[0]),
        "n_segments_poly": int(S_poly.shape[0]),
        "n_segments_graph": int(S_g.shape[0]),
        "W": float(W),
        "w_frame": float(w_frame) if add_frame else 0.0,
        "phi_target": float(phi_target),
        "phi_achieved": float(phi_ach),
        "phi_frame": float(phi_frame),
        "seed": seed,
        "add_frame": bool(add_frame),
        "frame_mode": "ring_outer_minus_inner",
        "frame_graph_edges_added": int(frame_edges_added),
        "graph_n": int(G.vcount()),
        "graph_m": int(G.ecount()),
        "merge_tol": float(merge_tol),
    }
    return polys, meta, G
