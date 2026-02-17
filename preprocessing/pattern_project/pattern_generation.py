import numpy as np
from shapely.geometry import Polygon,box, LineString
from scipy.spatial import Voronoi
from shapely.ops import unary_union
import igraph as ig


def _bar(p0, p1, w, dom=None):
    g = LineString([p0, p1]).buffer(w / 2, cap_style=2, join_style=2)
    return g if dom is None else g.intersection(dom)

def generate_vertical_struts(
    Lx, Ly, N=7, *, frame=True, W_frame=0.02
):
    """
    Build N vertical struts spanning full height, all with thickness = W_frame.

    Geometry outputs:
      - Polygons:
          * vertical bars (N of them), thickness=W_frame
          * if frame=True: exact frame ring = outer rect minus inner rect (thickness=W_frame)
          * if frame=False: add ONLY top+bottom boundary bars (thickness=W_frame)
            so you still get "up/down" polygons to weld the struts.

      - Graph:
          * vertical strut edges, thickness=W_frame
          * if frame=True: frame skeleton = INNER perimeter rectangle (offset by W_frame)
          * if frame=False: add ONLY top+bottom boundary polyline edges (on y=0 and y=Ly)

    Returns
    -------
    polys : list[shapely geometry]
    meta  : dict
    G     : igraph.Graph
    """

    Lx, Ly = float(Lx), float(Ly)
    W_frame = float(W_frame)
    dom = box(0.0, 0.0, Lx, Ly)
    A_dom = Lx * Ly

    xs = np.linspace(0, Lx, N + 2)[1:-1] if int(N) > 0 else np.array([])

    # shared vertices: bottom/top endpoints at strut x-positions + corners
    bottom = [(0.0, 0.0)] + [(float(x), 0.0) for x in xs] + [(Lx, 0.0)]
    top    = [(0.0, Ly)]  + [(float(x), Ly)  for x in xs] + [(Lx, Ly)]
    bottom.sort(key=lambda p: p[0])
    top.sort(key=lambda p: p[0])

    V = []
    for p in bottom + top:
        if p not in V:
            V.append(p)

    vid = {p: i for i, p in enumerate(V)}
    vx = [p[0] for p in V]
    vy = [p[1] for p in V]

    # exact frame ring polygon (optional)
    frame_poly = None
    inner_rect = None
    if frame and W_frame > 0:
        if (Lx - 2 * W_frame) <= 0 or (Ly - 2 * W_frame) <= 0:
            frame_poly = dom
            inner_rect = None
        else:
            inner_rect = box(W_frame, W_frame, Lx - W_frame, Ly - W_frame)
            frame_poly = dom.difference(inner_rect)

    edges, lengths, ths = [], [], []

    def add(p0, p1, wt):
        # p0/p1 can be new; add on demand
        if p0 not in vid:
            vid[p0] = len(vx)
            vx.append(float(p0[0])); vy.append(float(p0[1]))
        if p1 not in vid:
            vid[p1] = len(vx)
            vx.append(float(p1[0])); vy.append(float(p1[1]))
        u, v = vid[p0], vid[p1]
        if u == v:
            return
        edges.append((u, v))
        lengths.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        ths.append(float(wt))

    w = W_frame

    # vertical struts (full height)
    for x in xs:
        add((float(x), 0.0), (float(x), Ly), w)

    # frame graph
    if frame and (frame_poly is not None) and (inner_rect is not None):
        # INNER perimeter skeleton (consistent with your other generators)
        xmin, ymin = W_frame, W_frame
        xmax, ymax = Lx - W_frame, Ly - W_frame
        add((xmin, ymin), (xmax, ymin), w)
        add((xmax, ymin), (xmax, ymax), w)
        add((xmax, ymax), (xmin, ymax), w)
        add((xmin, ymax), (xmin, ymin), w)
    else:
        # boundary "up/down" only (so struts weld into top/bottom rails)
        for a, b in zip(bottom[:-1], bottom[1:]):
            add(a, b, w)
        for a, b in zip(top[:-1], top[1:]):
            add(a, b, w)

    # build graph
    G = ig.Graph(n=len(vx), edges=edges, directed=False)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"], G.es["thickness"] = lengths, ths

    # polygons: vertical bars + (frame ring) OR (top/bottom bars)
    polys = []
    for x in xs:
        polys.append(_bar((float(x), 0.0), (float(x), Ly), w, dom))

    if frame and (frame_poly is not None):
        polys.append(frame_poly)
    else:
        # explicit top/bottom polygons with thickness w (clipped to dom)
        polys.append(_bar((0.0, 0.0), (Lx, 0.0), w, dom))   # bottom rail
        polys.append(_bar((0.0, Ly),  (Lx, Ly),  w, dom))   # top rail

    polys = [p for p in polys if (p is not None and not p.is_empty)]

    if polys and A_dom > 0:
        u = unary_union(polys)
        A_fill = float(u.area)
        phi = A_fill / A_dom
    else:
        A_fill = 0.0
        phi = 0.0

    meta = dict(
        Lx=Lx, Ly=Ly, N=int(N), W_frame=W_frame, frame=bool(frame),
        phi=float(phi), A_fill=float(A_fill),
        graph_n=int(G.vcount()), graph_m=int(G.ecount()),
    )
    return polys, meta, G
def generate_vertical_struts_old(Lx, Ly, N=7, *, phi_target=0.05, frame=True, W_frame=0.02):
    """
    Vertical struts inside a rectangular frame.

    - Frame thickness is CONSTANT: W_frame
    - Only INNER strut thickness is adjusted to meet phi_target
    - Frame is included in BOTH the polygons and the graph
    """
    Lx, Ly = float(Lx), float(Ly)
    W_frame = float(W_frame)
    dom = box(0, 0, Lx, Ly)

    # inner strut x-locations (exclude boundaries)
    xs = np.linspace(0, Lx, N + 2)[1:-1] if N > 0 else np.array([])

    # ---- solve w_inner from area fraction phi_target ----
    A_dom = Lx * Ly
    if frame:
        # 4 bars, subtract the four W^2 corner overlaps
        A_frame = 2.0 * W_frame * (Lx + Ly) - 4.0 * (W_frame ** 2)
        A_frame = max(0.0, min(A_frame, A_dom))
    else:
        A_frame = 0.0

    A_target = float(phi_target) * A_dom
    A_inner_target = A_target - A_frame

    if N <= 0:
        w_inner = 0.0
    else:
        w_inner = A_inner_target / (N * Ly)

    # clamp to sane range (no negative thickness; no more than Lx)
    w_inner = max(0.0, min(w_inner, Lx))

    # ---- build vertices ----
    bottom = [(0.0, 0.0)] + [(float(x), 0.0) for x in xs] + [(Lx, 0.0)]
    top    = [(0.0, Ly)]  + [(float(x), Ly)  for x in xs] + [(Lx, Ly)]
    bottom.sort(key=lambda p: p[0])
    top.sort(key=lambda p: p[0])

    V = []
    for p in bottom + top:
        if p not in V:
            V.append(p)

    vid = {p: i for i, p in enumerate(V)}
    vx = [p[0] for p in V]
    vy = [p[1] for p in V]

    edges, lengths, ths = [], [], []

    def add(p0, p1, wt):
        u, v = vid[p0], vid[p1]
        if u == v:
            return
        edges.append((u, v))
        lengths.append(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        ths.append(float(wt))

    # inner vertical struts (thickness tuned)
    for x in xs:
        add((x, 0.0), (x, Ly), w_inner)

    # frame edges (constant thickness W_frame)
    if frame:
        for a, b in zip(bottom[:-1], bottom[1:]): add(a, b, W_frame)
        for a, b in zip(top[:-1], top[1:]):       add(a, b, W_frame)
        add((0.0, 0.0), (0.0, Ly), W_frame)
        add((Lx, 0.0),  (Lx, Ly), W_frame)

    # ---- graph ----
    G = ig.Graph(n=len(V), edges=edges)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"], G.es["thickness"] = lengths, ths

    # ---- polygons ----
    polys = []
    for x in xs:
        polys.append(_bar((x, 0.0), (x, Ly), w_inner, dom))

    if frame:
        polys += [
            _bar((0, 0), (Lx, 0), W_frame, dom),
            _bar((0, Ly), (Lx, Ly), W_frame, dom),
            _bar((0, 0), (0, Ly), W_frame, dom),
            _bar((Lx, 0), (Lx, Ly), W_frame, dom),
        ]

    polys = [p for p in polys if not p.is_empty]

    # recompute achieved phi from the (approx) areas we used
    A_inner = (N * w_inner * Ly) if N > 0 else 0.0
    phi_achieved = (A_frame + A_inner) / A_dom if A_dom > 0 else 0.0

    meta = dict(
        Lx=Lx, Ly=Ly, N=N,
        phi_target=float(phi_target),
        phi_achieved=float(phi_achieved),
        w_inner=float(w_inner),
        W_frame=W_frame if frame else 0.0,
        graph_n=G.vcount(), graph_m=G.ecount()
    )

    return polys, meta, G


def edge_rectangles(start_points, end_points, width):
    """
    Create Shapely rectangles (as thin polygons) representing bars/edges between
    start and end points, with the specified width.
    Vectorized for efficiency.
    """
    # Direction vectors
    directions = end_points - start_points
    lengths = np.hypot(directions[:, 0], directions[:, 1])
    # Unit vectors (edge direction)
    ux = directions[:, 0] / lengths
    uy = directions[:, 1] / lengths
    # Normals (for bar width)
    nx = -uy
    ny = ux
    half_width = width / 2
    offset = np.stack([nx, ny], axis=1) * half_width

    corners1 = start_points + offset
    corners2 = start_points - offset
    corners3 = end_points - offset
    corners4 = end_points + offset

    polygons = [
        Polygon([corners1[i], corners2[i], corners3[i], corners4[i]])
        for i in range(start_points.shape[0])
    ]
    return polygons

def edge_rectangles_per_width(starts, ends, widths):
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    widths = np.asarray(widths, float)

    polys = []
    for s, e, w in zip(starts, ends, widths):
        if w <= 0:
            polys.append(None)
            continue
        dx, dy = (e - s)
        L = float(np.hypot(dx, dy))
        if L <= 0:
            polys.append(None)
            continue

        # unit normal
        nx, ny = -dy / L, dx / L
        hx, hy = 0.5 * w * nx, 0.5 * w * ny  # half-width

        p0 = (s[0] + hx, s[1] + hy)
        p1 = (e[0] + hx, e[1] + hy)
        p2 = (e[0] - hx, e[1] - hy)
        p3 = (s[0] - hx, s[1] - hy)

        polys.append(Polygon([p0, p1, p2, p3]))

    return polys

def solve_inner_width_for_phi_frame_in_graph(
    starts, ends, is_frame_edge, Lx, Ly, phi_target, *,
    W_frame, tol=1e-4, max_iter=60, clip=True
):
    """
    starts/ends: all segments (inner + frame)
    is_frame_edge: (N,) bool array, True for frame segments
    W_frame fixed; solve for W_inner so union area fraction matches phi_target.
    """
    is_frame_edge = np.asarray(is_frame_edge, dtype=bool)
    if len(is_frame_edge) != len(starts):
        raise ValueError("is_frame_edge must match number of segments")

    if not (0.0 < phi_target < 1.0):
        raise ValueError("phi_target must be in (0,1).")

    def polys_for(W_inner):
        widths = np.where(is_frame_edge, float(W_frame), float(W_inner))
        polys_all = edge_rectangles_per_width(starts, ends, widths)
        if clip:
            dom = box(0.0, 0.0, Lx, Ly)
            polys_all = [g.intersection(dom) if g is not None else None for g in polys_all]
            polys_all = [g if (g is not None and (not g.is_empty) and g.area > 0) else None for g in polys_all]
        polys = [g for g in polys_all if g is not None]
        return polys, polys_all

    # Lower bound: inner=0 (frame only, plus any rails you might have marked as non-frame)
    polys0, polys0_all = polys_for(0.0)
    phi0 = area_fraction(polys0, Lx, Ly)
    if phi_target <= phi0 + tol:
        return 0.0, float(phi0), polys0, polys0_all

    W_lo = 0.0
    W_hi = min(Lx, Ly) * 1e-3

    for _ in range(100):
        polys_hi, polys_hi_all = polys_for(W_hi)
        phi_hi = area_fraction(polys_hi, Lx, Ly)
        if phi_hi >= phi_target:
            break
        W_hi *= 2.0
        if W_hi >= min(Lx, Ly) * 2.0:
            break

    polys_hi, polys_hi_all = polys_for(W_hi)
    phi_hi = area_fraction(polys_hi, Lx, Ly)
    if phi_hi < phi_target:
        return float(W_hi), float(phi_hi), polys_hi, polys_hi_all

    best_polys, best_phi, best_all = polys_hi, phi_hi, polys_hi_all
    for _ in range(max_iter):
        W_mid = 0.5 * (W_lo + W_hi)
        polys_mid, polys_mid_all = polys_for(W_mid)
        phi_mid = area_fraction(polys_mid, Lx, Ly)

        best_polys, best_phi, best_all = polys_mid, phi_mid, polys_mid_all

        if abs(phi_mid - phi_target) <= tol:
            return float(W_mid), float(phi_mid), polys_mid, polys_mid_all

        if phi_mid < phi_target:
            W_lo = W_mid
        else:
            W_hi = W_mid

    return float(0.5*(W_lo+W_hi)), float(best_phi), best_polys, best_all



def snap_pts_to_grid(pts, p, Lx, Ly, eps=1e-10):
    pts = np.asarray(pts, dtype=float).copy()

    # snap to boundaries first
    for col, L in [(0, Lx), (1, Ly)]:
        v = pts[:, col]
        v[np.isclose(v, 0.0, atol=eps)] = 0.0
        v[np.isclose(v, L,   atol=eps)] = float(L)
        pts[:, col] = v

    # snap interior to pitch grid (avoid moving boundary points you just fixed)
    # x
    mask_x = ~np.isclose(pts[:,0], 0.0, atol=eps) & ~np.isclose(pts[:,0], Lx, atol=eps)
    pts[mask_x, 0] = np.round(pts[mask_x, 0] / p) * p
    # y
    mask_y = ~np.isclose(pts[:,1], 0.0, atol=eps) & ~np.isclose(pts[:,1], Ly, atol=eps)
    pts[mask_y, 1] = np.round(pts[mask_y, 1] / p) * p

    return pts

def lattice_params_polys_and_graph(
    Lx, Ly, p, phi_target, *,
    W_frame=0.01,
    tol=1e-4, max_iter=60,
    clip=True,
    stitch_every_row=True,
):

    Lx = float(Lx); Ly = float(Ly); p = float(p); phi_target = float(phi_target)
    W_frame = float(W_frame)

    # --- y grid must be exact ---
    m_y = int(round(Ly / p))
    if abs(Ly / p - m_y) > 1e-9:
        raise ValueError(f"Need Ly/p integer. Got Ly/p={Ly/p:g}")
    y = np.arange(m_y + 1, dtype=float) * p
    y[-1] = Ly

    # --- x grid for inner lattice (may not reach Lx) ---
    nx = int(np.floor(Lx / p))
    x = np.arange(nx + 1, dtype=float) * p
    x[0] = 0.0
    x_last = float(x[-1])

    # --- inner lattice segments (square) ---
    xx, yy = np.meshgrid(x, y, indexing="ij")
    starts_inner = np.vstack([
        np.c_[xx[:-1, :].ravel(), yy[:-1, :].ravel()],   # horizontal
        np.c_[xx[:, :-1].ravel(), yy[:, :-1].ravel()],   # vertical
    ])
    ends_inner = np.vstack([
        np.c_[xx[1:,  :].ravel(), yy[1:,  :].ravel()],
        np.c_[xx[:,  1:].ravel(), yy[:,  1:].ravel()],
    ])

    # --- stitches fill the true gap x_last->Lx if needed ---
    if x_last < Lx - 1e-15:
        ys = y if stitch_every_row else np.array([0.0, Ly], float)
        starts_inner = np.vstack([starts_inner, np.c_[np.full(len(ys), x_last), ys]])
        ends_inner   = np.vstack([ends_inner,   np.c_[np.full(len(ys), Lx),     ys]])

    # --- frame segments on x-grid plus boundary Lx ---
    x_frame = x if abs(x_last - Lx) < 1e-15 else np.r_[x, Lx]
    fs, fe = boundary_frame_segments(x_frame, y, Lx, Ly)

    # --- all segments + frame mask ---
    starts = np.vstack([starts_inner, fs])
    ends   = np.vstack([ends_inner,   fe])
    is_frame = np.zeros(len(starts), dtype=bool)
    is_frame[len(starts_inner):] = True

    # --- solve W_inner with fixed frame ---
    W_inner, phi_star, _, _ = solve_inner_width_for_phi_frame_in_graph(
        starts, ends, is_frame, Lx, Ly, phi_target,
        W_frame=W_frame, tol=tol, max_iter=max_iter, clip=clip
    )

    widths = np.where(is_frame, W_frame, W_inner).astype(float)

    # --- polygons (aligned to segments) ---
    polys_all = edge_rectangles_per_width(starts, ends, widths)
    if clip:
        dom = box(0.0, 0.0, Lx, Ly)
        polys_all = [g.intersection(dom) if g is not None else None for g in polys_all]
    polys_all = [g if (g is not None and (not g.is_empty) and g.area > 0) else None for g in polys_all]
    polys = [g for g in polys_all if g is not None]

    # --- graph (grid-keyed, robust) ---
    vid, vx, vy = make_vid_from_grid(p, Lx, Ly)
    edges, lengths, thk, teff = [], [], [], []
    for s, e, w, poly in zip(starts, ends, widths, polys_all):
        u, v = vid(s), vid(e)
        if u == v:
            continue
        L = float(np.hypot(e[0]-s[0], e[1]-s[1]))
        if L <= 0:
            continue
        edges.append((u, v))
        lengths.append(L)
        thk.append(float(w))
        teff.append(float(w) if poly is None else float(poly.area / L))

    G = ig.Graph(n=len(vx), edges=edges, directed=False)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"] = lengths
    G.es["thickness"] = thk
    G.es["thickness_eff"] = teff

    meta = dict(
        p=p, m_y=m_y, nx=nx, x_last=x_last,
        Lx=Lx, Ly=Ly,
        W_inner=float(W_inner), W_frame=float(W_frame),
        phi_target=phi_target, phi_achieved=float(phi_star),
        n_segments=int(len(starts)),
        stitch_every_row=bool(stitch_every_row),
        clip=bool(clip),
    )
    return polys, meta, G

def boundary_frame_segments(x, y, Lx, Ly, inset):
    """
    Frame segments inset from the domain boundary by `inset` (typically W_frame/2),
    so the rectangles lie fully inside and clipping doesn't halve thickness.
    """
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    inset = float(inset)

    # clamp inset
    inset = max(0.0, min(inset, 0.49*min(Lx, Ly)))

    x0, x1 = inset, Lx - inset
    y0, y1 = inset, Ly - inset

    # For bottom/top, use x grid but clipped to [x0,x1] endpoints
    xb = x.copy()
    xb[0] = x0
    xb[-1] = x1

    yb = y.copy()
    yb[0] = y0
    yb[-1] = y1

    nx = len(xb) - 1
    ny = len(yb) - 1
    zeros_x = np.zeros(nx)
    zeros_y = np.zeros(ny)

    # bottom/top at y0/y1
    b0 = np.c_[xb[:-1], zeros_x + y0]; b1 = np.c_[xb[1:], zeros_x + y0]
    t0 = np.c_[xb[:-1], zeros_x + y1]; t1 = np.c_[xb[1:], zeros_x + y1]

    # left/right at x0/x1
    l0 = np.c_[zeros_y + x0, yb[:-1]]; l1 = np.c_[zeros_y + x0, yb[1:]]
    r0 = np.c_[zeros_y + x1, yb[:-1]]; r1 = np.c_[zeros_y + x1, yb[1:]]

    return np.vstack([b0, t0, l0, r0]), np.vstack([b1, t1, l1, r1])


def make_vid_from_grid(p, Lx, Ly, eps=1e-12):
    """Return a vertex-id function that keys points by (grid index, boundary flags)."""
    mp, vx, vy = {}, [], []

    def xk(x):
        if abs(x-0.0) < eps: return ("B", 0)
        if abs(x-Lx)  < eps: return ("B", 1)
        return ("I", int(round(x/p)))

    def yk(y):
        if abs(y-0.0) < eps: return ("B", 0)
        if abs(y-Ly)  < eps: return ("B", 1)
        return ("I", int(round(y/p)))

    def vid(pt):
        k = (xk(float(pt[0])), yk(float(pt[1])))
        j = mp.get(k)
        if j is None:
            j = len(vx); mp[k] = j
            vx.append(float(pt[0])); vy.append(float(pt[1]))
        return j

    return vid, vx, vy

def lattice_params_polys_and_graph(
    Lx, Ly, p, phi_target, *,
    W_frame=0.01,
    tol=1e-4, max_iter=60,
    clip=True,
    stitch_every_row=True,
):

    Lx = float(Lx); Ly = float(Ly); p = float(p); phi_target = float(phi_target)
    W_frame = float(W_frame)

    # --- y grid must be exact ---
    m_y = int(round(Ly / p))
    if abs(Ly / p - m_y) > 1e-9:
        raise ValueError(f"Need Ly/p integer. Got Ly/p={Ly/p:g}")
    y = np.arange(m_y + 1, dtype=float) * p
    y[-1] = Ly

    # --- x grid for inner lattice (may not reach Lx) ---
    nx = int(np.floor(Lx / p))
    x = np.arange(nx + 1, dtype=float) * p
    x[0] = 0.0
    x_last = float(x[-1])

    # --- inner lattice segments (square) ---
    xx, yy = np.meshgrid(x, y, indexing="ij")
    starts_inner = np.vstack([
        np.c_[xx[:-1, :].ravel(), yy[:-1, :].ravel()],   # horizontal
        np.c_[xx[:, :-1].ravel(), yy[:, :-1].ravel()],   # vertical
    ])
    ends_inner = np.vstack([
        np.c_[xx[1:,  :].ravel(), yy[1:,  :].ravel()],
        np.c_[xx[:,  1:].ravel(), yy[:,  1:].ravel()],
    ])

    # --- stitches fill the true gap x_last->Lx if needed ---
    if x_last < Lx - 1e-15:
        ys = y if stitch_every_row else np.array([0.0, Ly], float)
        starts_inner = np.vstack([starts_inner, np.c_[np.full(len(ys), x_last), ys]])
        ends_inner   = np.vstack([ends_inner,   np.c_[np.full(len(ys), Lx),     ys]])

    # --- frame segments on x-grid plus boundary Lx ---
    x_frame = x if abs(x_last - Lx) < 1e-15 else np.r_[x, Lx]
    fs, fe = boundary_frame_segments(x_frame, y, Lx, Ly, W_frame/2)

    # --- all segments + frame mask ---
    starts = np.vstack([starts_inner, fs])
    ends   = np.vstack([ends_inner,   fe])
    is_frame = np.zeros(len(starts), dtype=bool)
    is_frame[len(starts_inner):] = True

    # --- solve W_inner with fixed frame ---
    W_inner, phi_star, _, _ = solve_inner_width_for_phi_frame_in_graph(
        starts, ends, is_frame, Lx, Ly, phi_target,
        W_frame=W_frame, tol=tol, max_iter=max_iter, clip=clip
    )

    widths = np.where(is_frame, W_frame, W_inner).astype(float)

    # --- polygons (aligned to segments) ---
    polys_all = edge_rectangles_per_width(starts, ends, widths)
    if clip:
        dom = box(0.0, 0.0, Lx, Ly)
        polys_all = [g.intersection(dom) if g is not None else None for g in polys_all]
    polys_all = [g if (g is not None and (not g.is_empty) and g.area > 0) else None for g in polys_all]
    polys = [g for g in polys_all if g is not None]

    # frame polygons only
    frame_polys = [g for g, fr in zip(polys_all, is_frame) if fr and g is not None]
    inner_polys = [g for g, fr in zip(polys_all, is_frame) if (not fr) and g is not None]

    print("frame area fraction:", sum(g.area for g in frame_polys) / (Lx*Ly))
    print("inner area fraction:", sum(g.area for g in inner_polys) / (Lx*Ly))

    # --- graph (grid-keyed, robust) ---
    vid, vx, vy = make_vid_from_grid(p, Lx, Ly)
    edges, lengths, thk, teff = [], [], [], []
    for s, e, w, poly in zip(starts, ends, widths, polys_all):
        u, v = vid(s), vid(e)
        if u == v:
            continue
        L = float(np.hypot(e[0]-s[0], e[1]-s[1]))
        if L <= 0:
            continue
        edges.append((u, v))
        lengths.append(L)
        thk.append(float(w))
        teff.append(float(w) if poly is None else float(poly.area / L))

    G = ig.Graph(n=len(vx), edges=edges, directed=False)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"] = lengths
    G.es["thickness"] = thk
    G.es["thickness_eff"] = teff

    meta = dict(
        p=p, m_y=m_y, nx=nx, x_last=x_last,
        Lx=Lx, Ly=Ly,
        W_inner=float(W_inner), W_frame=float(W_frame),
        phi_target=phi_target, phi_achieved=float(phi_star),
        n_segments=int(len(starts)),
        stitch_every_row=bool(stitch_every_row),
        clip=bool(clip),
    )
    return polys, meta, G

def initialize_points_rect(n_points=400, Lx=16.0, Ly=1.0, seed=None):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.random(n_points) * Lx, rng.random(n_points) * Ly])


def voronoi_ridge_segments_clipped(points, Lx, Ly, extend_factor=10.0):
    """
    Return segment start/end points for Voronoi ridges. Infinite ridges are
    extended and later clipped (via polygon clipping).
    """
    vor = Voronoi(points)
    center = points.mean(axis=0)

    starts, ends = [], []

    for (p_i, p_j), v_idx in zip(vor.ridge_points, vor.ridge_vertices):
        v_idx = np.asarray(v_idx, dtype=int)

        # finite ridge
        if np.all(v_idx >= 0):
            a = vor.vertices[v_idx[0]]
            b = vor.vertices[v_idx[1]]
            starts.append(a); ends.append(b)
            continue

        # infinite ridge: one finite vertex
        finite = v_idx[v_idx >= 0]
        if finite.size != 1:
            continue
        v0 = vor.vertices[finite[0]]

        pi, pj = points[p_i], points[p_j]
        t = pj - pi
        t_norm = np.linalg.norm(t)
        if t_norm == 0:
            continue
        t = t / t_norm
        n = np.array([-t[1], t[0]])

        midpoint = 0.5 * (pi + pj)
        direction = n if np.dot(midpoint - center, n) > 0 else -n

        diag = np.hypot(Lx, Ly)
        far = v0 + direction * (extend_factor * diag)

        starts.append(v0); ends.append(far)

    return np.asarray(starts, float), np.asarray(ends, float), vor


def boundary_frame_polygons(Lx, Ly, W, factor=1.0):
    """
    Create 4 rectangles forming a border frame around the domain.
    factor=1.0 -> thickness = W
    factor=0.5 -> thickness = W/2
    """
    t = float(factor) * float(W)
    if t <= 0:
        return []
    # Build slightly oversized, then clipping will keep them inside [0,Lx]x[0,Ly]
    return [
        box(-t,     0.0,   0.0+t, Ly),   # left
        box(Lx-t,   0.0,   Lx+t,  Ly),   # right
        box(0.0,   -t,     Lx,    0.0+t),# bottom
        box(0.0,   Ly-t,   Lx,    Ly+t), # top
    ]


def polygons_from_segments_with_width(
    starts, ends, W, Lx, Ly, clip=True,
    add_frame=False, frame_factor=1.0
):
    """
    Build rectangles around Voronoi segments with width W, optionally add a
    bounding frame of thickness frame_factor*W, then (optionally) clip to domain.
    """
    polys = edge_rectangles(starts, ends, W)

    if add_frame:
        polys = polys + boundary_frame_polygons(Lx, Ly, W, factor=frame_factor)

    if not clip:
        return polys

    domain = box(0.0, 0.0, Lx, Ly)
    out = []
    for g in polys:
        if g.is_empty:
            continue
        gg = g.intersection(domain)
        if (not gg.is_empty) and gg.area > 0:
            out.append(gg)
    return out


def area_fraction(polys, Lx, Ly):
    if not polys:
        return 0.0
    return unary_union(polys).area / (Lx * Ly)


def solve_width_for_phi(
    starts, ends, Lx, Ly, phi_target,
    tol=1e-3, max_iter=30, clip=True,
    add_frame=False, frame_factor=1.0
):
    """
    Bisection on W using true union area fraction for (network + optional frame).
    Returns: W, phi_achieved, polys
    """
    if not (0.0 < phi_target < 1.0):
        raise ValueError("phi_target must be in (0,1).")

    W_lo = 0.0
    W_hi = min(Lx, Ly) * 1e-3

    # expand upper bound until target reached (or saturation)
    for _ in range(100):
        polys_hi = polygons_from_segments_with_width(
            starts, ends, W_hi, Lx, Ly, clip=clip,
            add_frame=add_frame, frame_factor=frame_factor
        )
        phi_hi = area_fraction(polys_hi, Lx, Ly)
        if phi_hi >= phi_target:
            break
        W_hi *= 2.0
        if W_hi >= min(Lx, Ly) * 2.0:
            break

    polys_hi = polygons_from_segments_with_width(
        starts, ends, W_hi, Lx, Ly, clip=clip,
        add_frame=add_frame, frame_factor=frame_factor
    )
    phi_hi = area_fraction(polys_hi, Lx, Ly)
    if phi_hi < phi_target:
        # best effort: cannot reach target with reasonable W
        return float(W_hi), float(phi_hi), polys_hi

    # bisection
    best_polys, best_phi = polys_hi, phi_hi
    for _ in range(max_iter):
        W_mid = 0.5 * (W_lo + W_hi)
        polys_mid = polygons_from_segments_with_width(
            starts, ends, W_mid, Lx, Ly, clip=clip,
            add_frame=add_frame, frame_factor=frame_factor
        )
        phi_mid = area_fraction(polys_mid, Lx, Ly)

        best_polys, best_phi = polys_mid, phi_mid

        if abs(phi_mid - phi_target) <= tol:
            return float(W_mid), float(phi_mid), polys_mid

        if phi_mid < phi_target:
            W_lo = W_mid
        else:
            W_hi = W_mid

    W = 0.5 * (W_lo + W_hi)
    return float(W), float(best_phi), best_polys

def _clip_segments_to_box(starts, ends, xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0):
    """Clip line segments to [xmin,xmax]x[ymin,ymax]. Returns arrays (S,E) of clipped segments."""
    dom = box(float(xmin), float(ymin), float(xmax), float(ymax))
    S_out, E_out = [], []

    for a, b in zip(starts, ends):
        seg = LineString([tuple(map(float, a)), tuple(map(float, b))])
        inter = seg.intersection(dom)
        if inter.is_empty:
            continue

        if inter.geom_type == "LineString":
            coords = list(inter.coords)
            if len(coords) >= 2:
                S_out.append(coords[0])
                E_out.append(coords[-1])
        elif inter.geom_type == "MultiLineString":
            for ls in inter.geoms:
                coords = list(ls.coords)
                if len(coords) >= 2:
                    S_out.append(coords[0])
                    E_out.append(coords[-1])

    if not S_out:
        return np.zeros((0, 2), float), np.zeros((0, 2), float)

    return np.asarray(S_out, float), np.asarray(E_out, float)


def _build_igraph_from_segments(starts, ends, thicknesses, *, merge_tol):
    """
    Build an undirected igraph from segments by merging nearly-identical endpoints.

    merge_tol is in the same length units as coordinates (e.g., ~1e-9*diag).
    """
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    thicknesses = np.asarray(thicknesses, float)
    if starts.shape != ends.shape or starts.shape[1] != 2:
        raise ValueError("starts/ends must be (M,2) arrays.")
    if thicknesses.shape[0] != starts.shape[0]:
        raise ValueError("thicknesses must have length M.")

    # Spatial hash by quantization (fast, deterministic)
    q = float(merge_tol)
    if q <= 0:
        raise ValueError("merge_tol must be > 0.")

    def key(pt):
        return (int(np.round(pt[0] / q)), int(np.round(pt[1] / q)))

    vid = {}
    xs, ys = [], []
    edges = []
    lengths = []
    ths = []

    def get_vid(pt):
        k = key(pt)
        if k in vid:
            return vid[k]
        i = len(xs)
        vid[k] = i
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
        return i

    for a, b, w in zip(starts, ends, thicknesses):
        u = get_vid(a)
        v = get_vid(b)
        if u == v:
            continue
        L = float(np.hypot(xs[v] - xs[u], ys[v] - ys[u]))
        if L <= 1e-12:
            continue
        edges.append((u, v))
        lengths.append(L)
        ths.append(float(w))

    G = ig.Graph(n=len(xs), edges=edges, directed=False)
    G.vs["x"] = xs
    G.vs["y"] = ys
    G.es["length"] = lengths
    G.es["thickness"] = ths
    return G

def build_segmented_frame_segments_from_endpoints(starts, ends, Lx, Ly, *, tol=None):
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    diag = float(np.hypot(Lx, Ly))
    if tol is None:
        tol = max(1e-9 * diag, 1e-12)

    P = np.vstack([starts, ends])
    x = P[:, 0]; y = P[:, 1]

    def uniq_sorted(vals):
        vals = np.asarray(vals, float)
        vals.sort()
        out = []
        for v in vals:
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return np.asarray(out, float)

    xb = uniq_sorted(np.r_[0.0, Lx, x[np.abs(y - 0.0) <= tol]])
    xt = uniq_sorted(np.r_[0.0, Lx, x[np.abs(y - Ly)  <= tol]])
    yl = uniq_sorted(np.r_[0.0, Ly, y[np.abs(x - 0.0) <= tol]])
    yr = uniq_sorted(np.r_[0.0, Ly, y[np.abs(x - Lx)  <= tol]])

    Sf, Ef = [], []
    for i in range(len(xb) - 1): Sf.append([xb[i], 0.0]); Ef.append([xb[i+1], 0.0])
    for i in range(len(yr) - 1): Sf.append([Lx, yr[i]]); Ef.append([Lx, yr[i+1]])
    for i in range(len(xt) - 1): Sf.append([xt[i], Ly]);  Ef.append([xt[i+1], Ly])
    for i in range(len(yl) - 1): Sf.append([0.0, yl[i]]); Ef.append([0.0, yl[i+1]])

    if not Sf:
        return np.zeros((0,2), float), np.zeros((0,2), float)
    return np.asarray(Sf, float), np.asarray(Ef, float)


def build_igraph_from_segments(starts, ends, thicknesses, *, merge_tol):
    """
    Single, safe graph builder for all patterns.
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

    grid = {}  # (ix,iy) -> [vid,...]
    xs, ys = [], []
    edges, lengths, ths = [], [], []

    def cell(p):
        return (int(np.floor(p[0] / q)), int(np.floor(p[1] / q)))

    def get_vid(p):
        c = cell(p)
        best = None
        best_d2 = q*q
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cc = (c[0] + dx, c[1] + dy)
                for vid in grid.get(cc, []):
                    d2 = (xs[vid] - p[0])**2 + (ys[vid] - p[1])**2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = vid
        if best is not None:
            return best

        vid = len(xs)
        xs.append(float(p[0])); ys.append(float(p[1]))
        grid.setdefault(c, []).append(vid)
        return vid

    for a, b, w in zip(starts, ends, th):
        u = get_vid(a)
        v = get_vid(b)
        if u == v:
            continue
        L = float(np.hypot(xs[v] - xs[u], ys[v] - ys[u]))
        if L <= 1e-12:
            continue
        edges.append((u, v))
        lengths.append(L)
        ths.append(float(w))

    G = ig.Graph(n=len(xs), edges=edges, directed=False)
    G.vs["x"] = xs
    G.vs["y"] = ys
    G.es["length"] = lengths
    G.es["thickness"] = ths
    return G


def merge_tol_from_segments(starts, ends, Lx, Ly):
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    L = np.hypot(ends[:,0]-starts[:,0], ends[:,1]-starts[:,1])
    L = L[np.isfinite(L) & (L > 0)]
    diag = float(np.hypot(Lx, Ly))
    if L.size == 0:
        return max(1e-12, 1e-8 * diag)
    L1 = float(np.percentile(L, 1.0))
    return max(1e-8 * diag, 1e-3 * L1)

def inner_network_poly(S_in, E_in, W_inner, clip_geom):
    # 1) build all centerlines
    lines = [LineString([tuple(a), tuple(b)]) for a, b in zip(S_in, E_in)]
    merged = unary_union(lines)  # merges collinear/touching lines into a proper network

    # 2) buffer ONCE (junctions handled cleanly)
    poly = merged.buffer(
        W_inner/2,
        cap_style=3,   # square ends (good for welding)
        join_style=3,  # bevel joins (prevents spikes)
    )

    # 3) clip to allowed region
    return poly.intersection(clip_geom)

def generate_random_voronoi_network(
    Lx=16.0, Ly=1.0, n_points=400, phi_target=0.05,
    seed=None, tol=1e-3, max_iter=30, clip=True,
    add_frame=True, frame_factor=1.0
):
    """
    Old Voronoi network style (no inner clipping, same _bar usage),
    but with a proper in-domain frame ring that corners correctly
    and connects naturally (by overlap) with the Voronoi bars.

    Frame thickness = Wf = frame_factor * W (same coupling as before),
    and W is solved so that total union area fraction matches phi_target.
    """
    Lx, Ly = float(Lx), float(Ly)
    phi_target = float(phi_target)
    frame_factor = float(frame_factor)

    outer = box(0.0, 0.0, Lx, Ly)
    A_dom = Lx * Ly

    points = initialize_points_rect(n_points=n_points, Lx=Lx, Ly=Ly, seed=seed)
    starts, ends, vor = voronoi_ridge_segments_clipped(points, Lx, Ly)

    # Clip centerlines to the outer domain (safe, keeps old behavior)
    if clip:
        S, E = _clip_segments_to_box(starts, ends, 0.0, 0.0, Lx, Ly)
    else:
        S = np.asarray(starts, float)
        E = np.asarray(ends, float)

    def build_polys(W):
        W = float(W)
        polys = []

        # Voronoi struts (same bar behavior as before)
        for a, b in zip(S, E):
            p = _bar(tuple(a), tuple(b), W, outer)
            if p is not None and not p.is_empty:
                polys.append(p)

        # Frame as an exact ring: outer \ inner (perfect corners, fully inside domain)
        if add_frame:
            Wf = frame_factor * W
            if Wf > 0:
                inner = box(Wf, Wf, Lx - Wf, Ly - Wf)
                # if inner collapses, frame is just the whole outer
                frame_poly = outer if (Lx - 2*Wf <= 0 or Ly - 2*Wf <= 0) else outer.difference(inner)
                polys.append(frame_poly)

        return polys

    def phi_of_W(W):
        polys = build_polys(W)
        if not polys:
            return 0.0, polys
        u = unary_union(polys)
        return float(u.area) / A_dom, polys

    # --- solve W by bisection (exact union area) ---
    lo = 0.0
    phi_lo, polys_lo = phi_of_W(lo)

    # If target is 0 or negative, trivial
    if phi_target <= phi_lo + tol:
        W = lo
        phi_ach, polys = phi_lo, polys_lo
    else:
        # Find an upper bound
        hi = 0.01 * min(Lx, Ly)
        hi = max(hi, 1e-6)
        phi_hi, polys_hi = phi_of_W(hi)

        # Expand until we exceed target or hit a hard cap
        hard_cap = 0.49 * min(Lx, Ly)  # frame would degenerate near 0.5*min dimension
        while phi_hi < phi_target and hi < hard_cap:
            hi *= 2.0
            phi_hi, polys_hi = phi_of_W(hi)

        if phi_hi < phi_target:
            # Unreachable within cap: return best effort
            W = hi
            phi_ach, polys = phi_hi, polys_hi
        else:
            W = None
            polys = None
            phi_ach = None
            for _ in range(int(max_iter)):
                mid = 0.5 * (lo + hi)
                phi_mid, polys_mid = phi_of_W(mid)

                W, phi_ach, polys = mid, phi_mid, polys_mid
                if abs(phi_mid - phi_target) <= tol:
                    break
                if phi_mid < phi_target:
                    lo = mid
                else:
                    hi = mid

    meta = {
        "Lx": float(Lx), "Ly": float(Ly),
        "n_points": int(n_points),
        "n_segments": int(S.shape[0]),
        "W": float(W),
        "phi_target": float(phi_target),
        "phi_achieved": float(phi_ach),
        "seed": seed,
        "add_frame": bool(add_frame),
        "frame_factor": float(frame_factor),
        "W_frame": float(frame_factor * W) if add_frame else 0.0,
        "frame_mode": "ring_outer_minus_inner",
    }
    return polys, meta, points, vor
def generate_random_voronoi_network_old(
    Lx=16.0, Ly=1.0, n_points=400, phi_target=0.05,
    seed=None, tol=1e-3, max_iter=30, clip=True,
    add_frame=True, frame_factor=1.0
):
    """
    End-to-end random Voronoi network + optional bounding frame.
    Solves W so that total filled area fraction matches phi_target.

    Returns:
      polys, meta, points, vor, G
    where G has:
      - vertex attrs: x, y
      - edge attrs: length, thickness
    """
    points = initialize_points_rect(n_points=n_points, Lx=Lx, Ly=Ly, seed=seed)
    starts, ends, vor = voronoi_ridge_segments_clipped(points, Lx, Ly)

    W, phi_ach, polys = solve_width_for_phi(
        starts, ends, Lx, Ly, phi_target,
        tol=tol, max_iter=max_iter, clip=clip,
        add_frame=add_frame, frame_factor=frame_factor
    )

    # --- build an *accurate* graph matching the clipped geometry ---
    # Clip centerline segments to the box (so G matches what polys represent)
    S_clip, E_clip = _clip_segments_to_box(starts, ends, Lx, Ly)
    seg_th = np.full(S_clip.shape[0], float(W), dtype=float)

    # Add the frame centerlines as 4 boundary segments (thickness may differ)
    if add_frame:
        Wf = float(frame_factor) * float(W)
        Sf, Ef = build_segmented_frame_segments_from_endpoints(S_clip, E_clip, Lx, Ly)
        S_clip = np.vstack([S_clip, Sf])
        E_clip = np.vstack([E_clip, Ef])
        seg_th = np.concatenate([seg_th, np.full(Sf.shape[0], Wf)])

        # pick a merging tolerance based on domain scale (safe default)
        diag = float(np.hypot(Lx, Ly))
        merge_tol = max(1e-12, 1e-9 * diag)
        
    merge_tol = merge_tol_from_segments(S_clip, E_clip, Lx, Ly)
    G = build_igraph_from_segments(S_clip, E_clip, seg_th, merge_tol=merge_tol)

    meta = {
        "Lx": float(Lx), "Ly": float(Ly),
        "n_points": int(n_points),
        "n_segments_raw": int(starts.shape[0]),
        "n_segments_clipped": int(S_clip.shape[0]),
        "W": float(W),
        "phi_target": float(phi_target),
        "phi_achieved": float(phi_ach),
        "seed": seed,
        "add_frame": bool(add_frame),
        "frame_factor": float(frame_factor),
        "graph_n": int(G.vcount()),
        "graph_m": int(G.ecount()),
        "merge_tol": float(merge_tol),
    }
    return polys, meta, G



def generate_random_voronoi_network_old(
    Lx=16.0, Ly=1.0, n_points=400, phi_target=0.05,
    seed=None, tol=1e-3, max_iter=30, clip=True,
    add_frame=True, frame_factor=1.0
):
    """
    End-to-end random Voronoi network + optional bounding frame.
    Solves W so that total filled area fraction matches phi_target.
    """
    points = initialize_points_rect(n_points=n_points, Lx=Lx, Ly=Ly, seed=seed)
    starts, ends, vor = voronoi_ridge_segments_clipped(points, Lx, Ly)

    W, phi_ach, polys = solve_width_for_phi(
        starts, ends, Lx, Ly, phi_target,
        tol=tol, max_iter=max_iter, clip=clip,
        add_frame=add_frame, frame_factor=frame_factor
    )

    meta = {
        "Lx": float(Lx), "Ly": float(Ly),
        "n_points": int(n_points),
        "n_segments": int(starts.shape[0]),
        "W": float(W),
        "phi_target": float(phi_target),
        "phi_achieved": float(phi_ach),
        "seed": seed,
        "add_frame": bool(add_frame),
        "frame_factor": float(frame_factor),
    }
    return polys, meta, points, vor

