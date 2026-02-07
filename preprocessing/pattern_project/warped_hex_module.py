"""
A generator for a graded triangular lattice (graded along x),
optionally locally-warped, converted to strut polygons + optional boundary frame,
and solved for width W to hit a target area fraction (phi).

Coordinate convention:
  Domain is [0, Lx] x [0, Ly] (NOT centered).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from shapely.geometry import box, Point, LineString
from pattern_project.pattern_generation import solve_width_for_phi

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
# Local organic displacement field (band-limited, swirl-like)
# ----------------------------
class LocalDisplacementField:
    def __init__(self, amp_frac=0.02, L_min=0.1, L_max=0.3, nmodes=16, seed=0):
        self.amp_frac = float(amp_frac)
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.nmodes = int(nmodes)

        rng = np.random.default_rng(seed)
        thetas = rng.uniform(0, 2 * np.pi, size=self.nmodes)
        Ls = np.exp(rng.uniform(np.log(self.L_min), np.log(self.L_max), size=self.nmodes))
        ks = 2 * np.pi / (Ls + 1e-15)

        self.kx = ks * np.cos(thetas)
        self.ky = ks * np.sin(thetas)
        self.phi = rng.uniform(0, 2 * np.pi, size=self.nmodes)

        w = rng.normal(size=self.nmodes)
        w /= (np.sqrt(np.mean(w**2)) + 1e-12)
        self.w = w

    def disp_dimless(self, x, y):
        arg = self.kx * x + self.ky * y + self.phi
        dpsi_dx = np.sum(self.w * np.cos(arg) * self.kx)
        dpsi_dy = np.sum(self.w * np.cos(arg) * self.ky)

        ux = dpsi_dy
        uy = -dpsi_dx

        k_typ = np.sqrt(np.mean(self.kx**2 + self.ky**2)) + 1e-12
        return ux / k_typ, uy / k_typ


def _smoothstep01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def boundary_fade_window(x, y, rect, fade_frac=0.12):
    minx, miny, maxx, maxy = rect.bounds
    W = maxx - minx
    H = maxy - miny
    t = float(fade_frac) * min(W, H)

    dl = x - minx
    dr = maxx - x
    db = y - miny
    dt_ = maxy - y
    d = min(dl, dr, db, dt_)
    return _smoothstep01(d / (t + 1e-15))


def remove_affine_drift(nodes0, nodes1):
    X = nodes0
    Y = nodes1
    M = np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])
    ax, *_ = np.linalg.lstsq(M, Y[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(M, Y[:, 1], rcond=None)
    pred = np.column_stack([M @ ax, M @ ay])
    return X + (Y - pred)


# ----------------------------
# Lattice + warp
# ----------------------------
def build_graded_tri_lattice_graph(
    rect,
    s_min, s_max,
    a0=1.0,
    grading_mode="power",
    p=3.0,
    side="right",
):
    """
    Triangular lattice built as columns along x; grading is along x:
      local spacing dy ~ a0 * s(x)
      dx ~ (sqrt(3)/2) * dy
    """
    minx, miny, maxx, maxy = rect.bounds
    sqrt3 = math.sqrt(3.0)

    nodes = []
    cols_y = []
    cols_idx = []

    x = minx
    col = 0
    while x <= maxx + 1e-12:
        s_col = float(grading_value(x, minx, maxx, s_min, s_max, mode=grading_mode, p=p, side=side))
        dy = a0 * s_col
        dx = (sqrt3 / 2.0) * dy

        y0 = miny + (0.5 * dy if (col % 2 == 1) else 0.0)
        ys = np.arange(y0, maxy + dy, dy)

        col_inds = []
        col_ys = []
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

    # vertical edges (within column)
    for idx in cols_idx:
        for k in range(len(idx) - 1):
            i, j = int(idx[k]), int(idx[k + 1])
            edges.add((min(i, j), max(i, j)))

    # diagonal edges (to next column)
    for c in range(len(cols_idx) - 1):
        y_this = cols_y[c]
        idx_this = cols_idx[c]
        y_next = cols_y[c + 1]
        idx_next = cols_idx[c + 1]

        dy_med = float(np.median(np.diff(y_this))) if len(y_this) >= 2 else 0.0
        for y, i in zip(y_this, idx_this):
            i = int(i)
            for target in (y - 0.5 * dy_med, y + 0.5 * dy_med):
                jpos = np.searchsorted(y_next, target)
                cands = []
                if 0 <= jpos < len(y_next):
                    cands.append(jpos)
                if 0 <= jpos - 1 < len(y_next):
                    cands.append(jpos - 1)
                if not cands:
                    continue
                jj = min(cands, key=lambda k: abs(y_next[k] - target))
                j = int(idx_next[jj])
                edges.add((min(i, j), max(i, j)))

    return nodes, sorted(edges)


def apply_local_organic(
    nodes,
    rect_for_s,
    rect_for_window,
    s_min, s_max, a0,
    field: LocalDisplacementField,
    grading_mode="power",
    p=3.0,
    side="right",
    fade_frac=0.12,
    jitter_frac=0.0,
    seed=0,
    remove_affine=True,
):
    """
    Warp nodes locally:
      displacement scale ~ amp_frac * h(x), where h(x)=a0*s(x)
      fade near boundary of rect_for_window
    """
    if grading_kwargs is None:
        grading_kwargs = {}

    rng = np.random.default_rng(seed)
    minx, _, maxx, _ = rect_for_s.bounds

    out = nodes.copy()
    for i, (x, y) in enumerate(nodes):
        s_loc = float(grading_value(x, minx, maxx, s_min, s_max, mode=grading_mode, p=p, side=side))
        h = a0 * s_loc

        w = boundary_fade_window(x, y, rect_for_window, fade_frac=fade_frac)
        ux, uy = field.disp_dimless(x, y)

        dx = w * field.amp_frac * h * ux
        dy = w * field.amp_frac * h * uy

        if jitter_frac > 0:
            dx += w * rng.normal(scale=jitter_frac * h)
            dy += w * rng.normal(scale=jitter_frac * h)

        out[i, 0] = x + dx
        out[i, 1] = y + dy

    if remove_affine:
        out = remove_affine_drift(nodes, out)
    return out


def edges_to_lines(clip_rect, nodes, edges):
    lines = []
    for i, j in edges:
        p0 = (float(nodes[i, 0]), float(nodes[i, 1]))
        p1 = (float(nodes[j, 0]), float(nodes[j, 1]))
        seg = LineString([p0, p1])
        inter = seg.intersection(clip_rect)
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
    # geometry
    Lx: float = 4.3
    Ly: float = 1.0
    pad_frac: float = 0.5

    # grading (length units)
    s_min: float = 0.05
    s_max: float = 0.30
    a0: float = 1.0
    grading_mode: str = "power"   # "linear" or "power"
    p: float = 3.0                # used only if grading_mode="power"
    side: str = "right"           # "left" or "right"

    # warp
    organic_amp_frac: float = 0.02
    L_min: float = 0.20
    L_max: float = 0.80
    nmodes: int = 18
    seed: int = 2
    fade_frac: float = 0.14
    jitter_frac: float = 0.0
    remove_affine: bool = True

    # area fraction solve
    phi_target: float = 0.15
    clip: bool = True
    add_frame: bool = True
    frame_factor: float = 0.5
    tol: float = 1e-5
    max_iter: int = 30

def generate_warped_lattice_polys(params: WarpedLatticeParams):
    """
    Returns:
      polys_final : list[Polygon]   (material polygons, already includes frame if enabled)
      W_star      : float           (strut width that achieves phi_target)
      phi_star    : float           (achieved area fraction)
      starts, ends: (N,2) arrays    (final clipped centerline segments used)
    """
    Lx, Ly = float(params.Lx), float(params.Ly)
    dom = make_rectangle(Lx, Ly)
    dom_pad = pad_rectangle(dom, pad_frac=float(params.pad_frac))

    # 1) graded lattice (on padded domain)
    nodes0, edges = build_graded_tri_lattice_graph(
        dom_pad,
        s_min=float(params.s_min),
        s_max=float(params.s_max),
        a0=float(params.a0),
        grading_mode=str(params.grading_mode),
        p=float(params.p),
        side=str(params.side),
    )

    # 2) optional warp
    nodes = nodes0
    if params.organic_amp_frac and params.organic_amp_frac > 0:
        field = LocalDisplacementField(
            amp_frac=float(params.organic_amp_frac),
            L_min=float(params.L_min),
            L_max=float(params.L_max),
            nmodes=int(params.nmodes),
            seed=int(params.seed),
        )
        nodes = apply_local_organic(
            nodes0,
            rect_for_s=dom_pad,
            rect_for_window=dom,
            s_min=float(params.s_min),
            s_max=float(params.s_max),
            a0=float(params.a0),
            field=field,
            grading_mode=str(params.grading_mode),
            p=float(params.p),
            side=str(params.side),
            fade_frac=float(params.fade_frac),
            jitter_frac=float(params.jitter_frac),
            seed=int(params.seed) + 101,
            remove_affine=bool(params.remove_affine),
        )
    # 3) clip to original domain
    centerlines = edges_to_lines(dom, nodes, edges)
    starts = np.asarray([ln.coords[0] for ln in centerlines], dtype=float)
    ends = np.asarray([ln.coords[-1] for ln in centerlines], dtype=float)

    # 4) solve width for target phi using helper bisection
    W_star, phi_star, polys_final = solve_width_for_phi(
        starts, ends,
        Lx=Lx, Ly=Ly,
        phi_target=float(params.phi_target),
        tol=float(params.tol),
        max_iter=int(params.max_iter),
        clip=bool(params.clip),
        add_frame=bool(params.add_frame),
        frame_factor=float(params.frame_factor),
    )

    meta = {
        "Lx": float(Lx), "Ly": float(Ly),
        "mode": str(params.grading_mode),
        "s_min": float(params.s_min),
        "s_max": float(params.s_max),
        "a0": float(params.a0),
        "p": float(params.p),
        "side": str(params.side),
        "warp enabled": bool(params.organic_amp_frac and params.organic_amp_frac > 0),
        "polys_out": int(len(polys_final)) if polys_final is not None else 0,
        "W": float(W_star),
        }
    return polys_final, meta