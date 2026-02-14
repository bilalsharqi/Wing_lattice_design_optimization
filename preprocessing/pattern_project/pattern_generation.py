import numpy as np
from shapely.geometry import Polygon,box, LineString
from scipy.spatial import Voronoi
from shapely.ops import unary_union
import igraph as ig


def _bar(p0, p1, w, dom=None):
    g = LineString([p0, p1]).buffer(w / 2, cap_style=2, join_style=2)
    return g if dom is None else g.intersection(dom)


def generate_vertical_struts(Lx, Ly, N=7, w=0.02, *, frame=True, w_frame=None):
    Lx, Ly, w = float(Lx), float(Ly), float(w)
    w_frame = w if w_frame is None else float(w_frame)
    dom = box(0, 0, Lx, Ly)

    xs = np.linspace(0, Lx, N + 2)[1:-1] if N > 0 else np.array([])

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
        ths.append(wt)

    for x in xs:
        add((x, 0.0), (x, Ly), w)

    if frame:
        for a, b in zip(bottom[:-1], bottom[1:]): add(a, b, w_frame)
        for a, b in zip(top[:-1], top[1:]):       add(a, b, w_frame)
        add((0.0, 0.0), (0.0, Ly), w_frame)
        add((Lx, 0.0),  (Lx, Ly), w_frame)

    G = ig.Graph(n=len(V), edges=edges)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"], G.es["thickness"] = lengths, ths

    polys = []
    for x in xs:
        polys.append(_bar((x, 0.0), (x, Ly), w, dom))
    if frame:
        polys += [
            _bar((0, 0), (Lx, 0), w_frame, dom),
            _bar((0, Ly), (Lx, Ly), w_frame, dom),
            _bar((0, 0), (0, Ly), w_frame, dom),
            _bar((Lx, 0), (Lx, Ly), w_frame, dom),
        ]

    polys = [p for p in polys if not p.is_empty]

    meta = dict(Lx=Lx, Ly=Ly, N=N, w=w, w_frame=w_frame,
                graph_n=G.vcount(), graph_m=G.ecount())

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

def lattice_params_polys_and_graph(
    Lx, Ly, p, phi_target, *,
    tol=1e-4, max_iter=60,
    clip=True,
    stitch_every_row=True,   # if False, stitches only at y=0 and y=Ly
):
    Lx = float(Lx); Ly = float(Ly); p = float(p); phi_target = float(phi_target)

    # --- enforce exact pitch vertically ---
    m_y_float = Ly / p
    m_y = int(round(m_y_float))
    if abs(m_y_float - m_y) > 1e-9:
        raise ValueError(f"Need Ly/p integer for exact pitch. Got Ly/p={m_y_float:g}")
    Ny = m_y + 1
    y = np.arange(Ny) * p  # 0..Ly exactly

    # --- square lattice columns that fit in x ---
    nx = int(np.floor(Lx / p))
    Nx = nx + 1
    x = np.arange(Nx) * p              # 0..x_last
    x_last = float(x[-1])

    # --- lattice segments ---
    xx, yy = np.meshgrid(x, y, indexing="ij")
    h0 = np.c_[xx[:-1, :].ravel(), yy[:-1, :].ravel()]
    h1 = np.c_[xx[1:,  :].ravel(), yy[1:,  :].ravel()]
    v0 = np.c_[xx[:, :-1].ravel(), yy[:, :-1].ravel()]
    v1 = np.c_[xx[:,  1:].ravel(), yy[:,  1:].ravel()]
    starts = np.vstack([h0, v0])
    ends   = np.vstack([h1, v1])

    # --- left & right rails (vertical, split by pitch) ---
    left_starts  = np.c_[np.zeros(m_y), y[:-1]]
    left_ends    = np.c_[np.zeros(m_y), y[1:]]
    right_starts = np.c_[np.full(m_y, Lx), y[:-1]]
    right_ends   = np.c_[np.full(m_y, Lx), y[1:]]

    # --- stitches to prevent "floating" right rail ---
    if stitch_every_row:
        ys = y
    else:
        ys = np.array([0.0, Ly], dtype=float)

    # connect last lattice column (x_last) to boundary (Lx) at chosen y's
    stitch_starts = np.c_[np.full(len(ys), x_last), ys]
    stitch_ends   = np.c_[np.full(len(ys), Lx),     ys]

    # (optional) if you ever center the lattice, add left stitches too; here x starts at 0 so not needed.

    starts = np.vstack([starts, left_starts, right_starts, stitch_starts])
    ends   = np.vstack([ends,   left_ends,   right_ends,   stitch_ends])

    # --- solve thickness INCLUDING rails + stitches (they affect area) ---
    W_star, phi_star, _ = solve_width_for_phi(
        starts, ends,
        Lx=Lx, Ly=Ly,
        phi_target=float(phi_target),
        tol=float(tol),
        max_iter=int(max_iter),
        clip=bool(clip),
        add_frame=False,
        frame_factor=1.0,
    )
    W_star = float(W_star)
    phi_star = float(phi_star)

    # --- polygons ---
    polys_all = edge_rectangles(starts, ends, W_star)
    if clip:
        dom = box(0.0, 0.0, Lx, Ly)
        polys_all = [g.intersection(dom) for g in polys_all]
        polys_all = [g if (not g.is_empty and g.area > 0) else None for g in polys_all]
    else:
        polys_all = [g if (not g.is_empty and g.area > 0) else None for g in polys_all]
    polys = [g for g in polys_all if g is not None]

    # --- graph from segment endpoints (dedup by rounding) ---
    pts = np.vstack([starts, ends])
    scale = 1.0 / 1e-12
    keys = np.round(pts * scale).astype(np.int64)

    mp = {}
    vx, vy = [], []
    def vid(k, pxy):
        kt = (int(k[0]), int(k[1]))
        j = mp.get(kt)
        if j is None:
            j = len(vx)
            mp[kt] = j
            vx.append(float(pxy[0]))
            vy.append(float(pxy[1]))
        return j

    edges = []
    lengths = []
    nseg = len(starts)
    for s, e, ks, ke in zip(starts, ends, keys[:nseg], keys[nseg:]):
        u = vid(ks, s)
        v = vid(ke, e)
        if u == v:
            continue
        edges.append((u, v))
        lengths.append(float(np.hypot(e[0]-s[0], e[1]-s[1])))

    G = ig.Graph(n=len(vx), edges=edges, directed=False)
    G.vs["x"] = vx
    G.vs["y"] = vy
    G.es["length"] = lengths
    G.es["thickness"] = [W_star] * len(edges)

    # thickness_eff (aligned to segments; skip degenerates consistently)
    seg_len = np.hypot(ends[:, 0] - starts[:, 0], ends[:, 1] - starts[:, 1])
    seg_teff = [
        float(W_star) if (poly is None or L <= 0) else float(poly.area / float(L))
        for poly, L in zip(polys_all, seg_len)
    ]
    teff_kept = [te for L, te in zip(seg_len, seg_teff) if L > 0]
    G.es["thickness_eff"] = teff_kept

    meta = dict(
        p=float(p), m_y=int(m_y),
        nx=int(nx), x_last=float(x_last),
        Lx=float(Lx), Ly=float(Ly),
        W=float(W_star),
        phi_target=float(phi_target),
        phi_achieved=float(phi_star),
        n_segments=int(len(starts)),
        stitch_every_row=bool(stitch_every_row),
        clip=bool(clip),
    )
    return polys, meta, G

def lattice_params_polys_and_graph_old(Lx, Ly, p, phi_target, *, center=True, clip=True):
    """
    Finite-size explicit model (valid up to saturation W>=p):
    - Chooses Nx,Ny to fit inside Lx,Ly with square pitch p (allows leftover margins)
    - Computes W from phi_target using the finite-size strip-union model
    - Output polygons (bars) using your edge_rectangles so you can plot

    Returns:
      polys : list[shapely.geometry.Polygon]
      meta  : dict with Nx, Ny, p, W, phi_max, offsets, achieved_phi_model
        G:
      - vertex attrs: x, y
      - edge attrs: length, thickness (nominal bulk = W), thickness_eff (from polygon, may differ near boundary)
    """

    # ---- choose Nx,Ny + offsets ----
    Nx = int(np.floor(Lx / p)) + 1
    Ny = int(np.floor(Ly / p)) + 1
    if Nx < 2 or Ny < 2:
        raise ValueError("Pitch too large: need Nx,Ny >= 2.")

    Lx_lat = (Nx - 1) * p
    Ly_lat = (Ny - 1) * p
    ox = 0.5 * (Lx - Lx_lat) if center else 0.0
    oy = 0.5 * (Ly - Ly_lat) if center else 0.0

    # ---- invert finite-size phi(W) mapping ----
    phi_max = (Lx_lat * Ly_lat) / (Lx * Ly)
    if phi_target >= phi_max:
        W = float(p)
    else:
        Astar = phi_target * (Lx * Ly)
        B = (Nx - 1) * Ly_lat + (Ny - 1) * Lx_lat
        C = (Nx - 1) * (Ny - 1)
        disc = B * B - 4.0 * C * Astar
        if disc < 0:
            raise ValueError("Target phi unattainable for this (p,Nx,Ny) in W<p regime.")
        W = float((B - np.sqrt(disc)) / (2.0 * C))

    # achieved phi (same model)
    if W < p:
        lx, ly = (Nx - 1) * W, (Ny - 1) * W
    else:
        lx, ly = Lx_lat, Ly_lat
    phi_model = (lx * Ly_lat + ly * Lx_lat - lx * ly) / (Lx * Ly)

    # ---- node coords ----
    x = ox + np.arange(Nx) * p
    y = oy + np.arange(Ny) * p
    xx, yy = np.meshgrid(x, y, indexing="ij")
    X = xx.ravel()
    Y = yy.ravel()
    N = Nx * Ny

    # ---- edges (same ordering as polygon generation) ----
    # node id: id(i,j)=i*Ny+j
    eh = [(i*Ny + j, (i+1)*Ny + j) for i in range(Nx-1) for j in range(Ny)]
    ev = [(i*Ny + j, i*Ny + (j+1)) for i in range(Nx) for j in range(Ny-1)]
    edges = eh + ev

    # ---- graph ----
    G = ig.Graph(n=N, edges=edges, directed=False)
    G.vs["x"] = X.tolist()
    G.vs["y"] = Y.tolist()

    # lengths (robust even if p changes later)
    lengths = []
    for u, v in edges:
        lengths.append(float(np.hypot(X[v] - X[u], Y[v] - Y[u])))
    G.es["length"] = lengths
    G.es["thickness"] = [float(W)] * len(edges)  # bulk/nominal for all edges

    # ---- polygons in the same order as edges ----
    # (horizontal then vertical)
    h0 = np.c_[xx[:-1, :].ravel(), yy[:-1, :].ravel()]
    h1 = np.c_[xx[1:,  :].ravel(), yy[1:,  :].ravel()]
    v0 = np.c_[xx[:, :-1].ravel(), yy[:, :-1].ravel()]
    v1 = np.c_[xx[:,  1:].ravel(), yy[:,  1:].ravel()]
    polys_all = edge_rectangles(h0, h1, W) + edge_rectangles(v0, v1, W)

    # clip polygons (optional), but keep alignment to compute thickness_eff
    if clip:
        dom = box(0.0, 0.0, Lx, Ly)
        polys_all = [g.intersection(dom) for g in polys_all]
        polys_all = [g if (not g.is_empty and g.area > 0) else None for g in polys_all]

    # effective thickness from polygon: area / centerline length
    thickness_eff = []
    for poly, L in zip(polys_all, lengths):
        thickness_eff.append(float(W) if (poly is None or L <= 0) else float(poly.area / L))
    G.es["thickness_eff"] = thickness_eff

    # final polys list (drop empties for downstream plotting if you want)
    polys = [g for g in polys_all if g is not None]

    meta = dict(
        Nx=Nx, Ny=Ny, p=float(p), W=float(W),
        phi_target=float(phi_target), phi_model=float(phi_model), phi_max=float(phi_max),
        Lx_lat=float(Lx_lat), Ly_lat=float(Ly_lat), ox=float(ox), oy=float(oy),
        n_vertices=int(N), n_edges=int(len(edges)), clip=bool(clip),
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

def _clip_segments_to_box(starts, ends, Lx, Ly):
    """Clip line segments to [0,Lx]x[0,Ly]. Returns arrays (S,E) of clipped segments."""
    dom = box(0.0, 0.0, float(Lx), float(Ly))
    S_out, E_out = [], []

    for a, b in zip(starts, ends):
        seg = LineString([tuple(map(float, a)), tuple(map(float, b))])
        inter = seg.intersection(dom)
        if inter.is_empty:
            continue

        # intersection can be LineString or MultiLineString (rare but possible)
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


def generate_random_voronoi_network(
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

