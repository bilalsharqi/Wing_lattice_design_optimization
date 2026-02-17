import numpy as np
import igraph as ig
from shapely.geometry import LineString, box
from shapely.ops import unary_union


def frame_ring_polygon(Lx, Ly, W):
    """Exact frame material reaching the four corners, thickness=W."""
    Lx, Ly, W = float(Lx), float(Ly), float(W)
    if W <= 0:
        return None
    if 2 * W >= min(Lx, Ly):
        return box(0.0, 0.0, Lx, Ly)
    return box(0.0, 0.0, Lx, Ly).difference(box(W, W, Lx - W, Ly - W))


def polys_from_segments_buffer(starts, ends, w):
    """Square caps + miter joins; w can be scalar or per-segment array."""
    starts = np.asarray(starts, float)
    ends   = np.asarray(ends, float)
    w = np.asarray(w, float)
    if w.ndim == 0:
        w = np.full(len(starts), float(w), dtype=float)

    out = []
    for s, e, wi in zip(starts, ends, w):
        if wi <= 0:
            out.append(None); continue
        g = LineString([tuple(s), tuple(e)]).buffer(0.5 * float(wi), cap_style=2, join_style=2)
        out.append(g if (not g.is_empty and g.area > 0) else None)
    return out


def area_fraction(polys, Lx, Ly):
    polys = [g for g in polys if g is not None and (not g.is_empty) and g.area > 0]
    return 0.0 if not polys else unary_union(polys).area / (float(Lx) * float(Ly))


def clip_segments(starts, ends, dom_poly):
    out_s, out_e = [], []
    for s, e in zip(starts, ends):
        g = LineString([tuple(s), tuple(e)]).intersection(dom_poly)
        if g.is_empty:
            continue
        if g.geom_type == "LineString":
            a, b = g.coords[0], g.coords[-1]
            if a != b:
                out_s.append(a); out_e.append(b)
        else:  # MultiLineString
            for ls in g.geoms:
                a, b = ls.coords[0], ls.coords[-1]
                if a != b:
                    out_s.append(a); out_e.append(b)
    return np.asarray(out_s, float), np.asarray(out_e, float)


def snapper(p, Lx, Ly, x_planes=(), y_planes=(), eps=1e-9):
    p, Lx, Ly = float(p), float(Lx), float(Ly)
    x_planes = [0.0, Lx, *map(float, x_planes)]
    y_planes = [0.0, Ly, *map(float, y_planes)]

    def snap1(v, planes):
        for a in planes:
            if abs(v - a) < eps:
                return a
        return round(v / p) * p

    def snap_pt(pt):
        x = snap1(float(pt[0]), x_planes)
        y = snap1(float(pt[1]), y_planes)
        return (min(max(x, 0.0), Lx), min(max(y, 0.0), Ly))

    return snap_pt


def build_graph(starts, ends, snap_pt):
    mp, vx, vy = {}, [], []

    def vid(pt):
        k = snap_pt(pt)
        j = mp.get(k)
        if j is None:
            j = len(vx); mp[k] = j
            vx.append(k[0]); vy.append(k[1])
        return j

    edges, lengths = [], []
    for s, e in zip(starts, ends):
        u, v = vid(s), vid(e)
        if u == v:
            continue
        L = float(np.hypot(e[0] - s[0], e[1] - s[1]))
        if L > 0:
            edges.append((u, v))
            lengths.append(L)

    G = ig.Graph(n=len(vx), edges=edges, directed=False)
    G.vs["x"], G.vs["y"] = vx, vy
    G.es["length"] = lengths
    return G


def square_grid_frame_inside(
    Lx, Ly, p, phi_target, *,
    W_frame=0.01,
    extra_x=1.2,
    tol=1e-4, max_iter=60
):
    Lx, Ly, p = float(Lx), float(Ly), float(p)
    phi_target, W_frame = float(phi_target), float(W_frame)

    inset_f = 0.5 * W_frame
    frame_box = box(inset_f, inset_f, Lx - inset_f, Ly - inset_f)
    frame_poly = frame_ring_polygon(Lx, Ly, W_frame)

    # frame segments on inset rectangle, split by pitch
    xs = np.arange(inset_f, (Lx - inset_f) + 0.5 * p, p); xs[-1] = Lx - inset_f
    ys = np.arange(inset_f, (Ly - inset_f) + 0.5 * p, p); ys[-1] = Ly - inset_f
    fs = np.vstack([
        np.c_[xs[:-1], 0*xs[:-1] + inset_f],
        np.c_[xs[:-1], 0*xs[:-1] + (Ly - inset_f)],
        np.c_[0*ys[:-1] + inset_f, ys[:-1]],
        np.c_[0*ys[:-1] + (Lx - inset_f), ys[:-1]],
    ])
    fe = np.vstack([
        np.c_[xs[1:], 0*xs[1:] + inset_f],
        np.c_[xs[1:], 0*xs[1:] + (Ly - inset_f)],
        np.c_[0*ys[1:] + inset_f, ys[1:]],
        np.c_[0*ys[1:] + (Lx - inset_f), ys[1:]],
    ])

    # centered grid segments on extended x-range
    Lxg = extra_x * Lx
    x0, x1 = 0.5 * (Lx - Lxg), 0.5 * (Lx + Lxg)
    xg = np.arange(np.floor(x0 / p), np.ceil(x1 / p) + 1) * p
    yg = np.arange(0, int(round(Ly / p)) + 1) * p
    yg = yg.astype(float); yg[-1] = Ly

    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    gs = np.vstack([np.c_[xx[:-1, :].ravel(), yy[:-1, :].ravel()],
                    np.c_[xx[:, :-1].ravel(), yy[:, :-1].ravel()]])
    ge = np.vstack([np.c_[xx[1:,  :].ravel(), yy[1:,  :].ravel()],
                    np.c_[xx[:,  1:].ravel(), yy[:,  1:].ravel()]])

    # clip grid to frame interior
    gs, ge = clip_segments(gs, ge, frame_box)

    starts_all = np.vstack([gs, fs])
    ends_all   = np.vstack([ge, fe])

    def phi_for(W_inner):
        inner = polys_from_segments_buffer(gs, ge, float(W_inner))
        inner = [g for g in inner if g is not None and (not g.is_empty) and g.area > 0]
        polys = ([frame_poly] if frame_poly is not None else []) + inner
        return area_fraction(polys, Lx, Ly), polys

    # bisection on W_inner
    phi0, polys0 = phi_for(0.0)
    if phi_target <= phi0 + tol:
        W_inner, phi_star, polys = 0.0, phi0, polys0
    else:
        W_lo, W_hi = 0.0, min(Lx, Ly) * 1e-3
        for _ in range(80):
            phi_hi, _ = phi_for(W_hi)
            if phi_hi >= phi_target:
                break
            W_hi *= 2.0

        W_inner, phi_star, polys = W_hi, phi_hi, None
        for _ in range(int(max_iter)):
            W_mid = 0.5 * (W_lo + W_hi)
            phi_mid, polys_mid = phi_for(W_mid)
            W_inner, phi_star, polys = W_mid, phi_mid, polys_mid
            if abs(phi_mid - phi_target) <= tol:
                break
            if phi_mid < phi_target:
                W_lo = W_mid
            else:
                W_hi = W_mid

    # graph: snap to frame & inner planes, then pitch
    inset_i = 0.5 * float(W_inner)
    snap_pt = snapper(
        p, Lx, Ly,
        x_planes=(inset_f, Lx - inset_f, inset_i, Lx - inset_i),
        y_planes=(inset_f, Ly - inset_f, inset_i, Ly - inset_i),
        eps=1e-9
    )
    G = build_graph(starts_all, ends_all, snap_pt)

    # thickness per edge (frame planes)
    xs_v = np.asarray(G.vs["x"], float)
    ys_v = np.asarray(G.vs["y"], float)
    thk = np.full(G.ecount(), float(W_inner), float)
    is_frame = np.zeros(G.ecount(), bool)
    xL, xR = inset_f, (Lx - inset_f)
    yB, yT = inset_f, (Ly - inset_f)
    for ei, (u, v) in enumerate(G.get_edgelist()):
        on_frame = (
            (abs(xs_v[u]-xL)<1e-9 and abs(xs_v[v]-xL)<1e-9) or
            (abs(xs_v[u]-xR)<1e-9 and abs(xs_v[v]-xR)<1e-9) or
            (abs(ys_v[u]-yB)<1e-9 and abs(ys_v[v]-yB)<1e-9) or
            (abs(ys_v[u]-yT)<1e-9 and abs(ys_v[v]-yT)<1e-9)
        )
        if on_frame:
            thk[ei] = W_frame
            is_frame[ei] = True
    G.es["thickness"] = thk.tolist()
    G.es["is_frame"] = is_frame.tolist()

    meta = dict(
        Lx=Lx, Ly=Ly, p=p, extra_x=float(extra_x),
        W_frame=W_frame, W_inner=float(W_inner),
        phi_target=float(phi_target), phi_achieved=float(phi_star),
        n_segments=int(len(starts_all)),
    )
    return polys, meta, G
