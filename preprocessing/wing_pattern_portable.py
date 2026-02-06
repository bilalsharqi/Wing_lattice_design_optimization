from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union
from typing import Iterable, Tuple, Dict, List, Union
import gmsh
from shapely.geometry.polygon import orient

# ---------- helpers ----------

def _iter_polys(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if isinstance(g, GeometryCollection):
        return [p for gg in g.geoms for p in _iter_polys(gg)]
    return []


def _snap_ring(coords, tol):
    pts = np.asarray(coords, float)

    if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=max(tol, 1e-15)):
        pts = pts[:-1]

    if tol > 0:
        pts = np.round(pts / tol) * tol

    keep = [0]
    for i in range(1, len(pts)):
        if not np.allclose(pts[i], pts[keep[-1]], atol=max(tol, 1e-15)):
            keep.append(i)

    pts = pts[keep]
    return [(float(x), float(y)) for x, y in pts] if len(pts) >= 3 else []


def _preprocess(polygons, *, buffer0, simplify):
    out = []
    for g in polygons:
        for p in _iter_polys(g):
            if p.is_empty or p.area <= 0:
                continue
            if buffer0:
                p = p.buffer(0)
            if simplify > 0:
                p = p.simplify(simplify, preserve_topology=True)
            out.append(orient(p, 1.0))
    return out


# ---------- main exporter ----------
def export_polygons_to_step(
    polygons: Iterable[Polygon],
    z_thickness: float,
    step_path: str,
    *,
    model_name: str = "poly_union",
    tol_snap: float = 1e-6,
    simplify_tol: float = 0.0,
    per_poly_buffer0: bool = True,
    occ_fix: bool = True,
) -> None:
    """
    Polygons → 2D union → 2D cut (holes) → single extrusion → STEP
    """

    # ---- 2D geometry prep (Shapely) ----
    prepped = _preprocess(
        polygons,
        buffer0=per_poly_buffer0,
        simplify=simplify_tol,
    )
    poly = orient(unary_union(prepped), 1.0)

    # ---- Gmsh / OCC session ----
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add(model_name)

    if occ_fix:
        for opt in (
            "Geometry.OCCFixDegenerated",
            "Geometry.OCCFixSmallEdges",
            "Geometry.OCCFixSmallFaces",
        ):
            gmsh.option.setNumber(opt, 1)

    # ---- vertex handling ----
    inv = 1.0 / tol_snap if tol_snap > 0 else None
    cache: Dict[Tuple[int, int], int] = {}

    def add_point(x: float, y: float) -> int:
        if inv is None:
            return gmsh.model.occ.addPoint(x, y, 0.0)
        key = (int(round(x * inv)), int(round(y * inv)))
        if key not in cache:
            cache[key] = gmsh.model.occ.addPoint(x, y, 0.0)
        return cache[key]

    def make_loop(coords) -> int:
        ring = _snap_ring(coords, tol_snap)
        pts = [add_point(x, y) for x, y in ring]
        lines = [
            gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
            for i in range(len(pts))
        ]
        return gmsh.model.occ.addCurveLoop(lines)

    # ---- build 2D region: outer minus holes ----
    outer = gmsh.model.occ.addPlaneSurface(
        [make_loop(poly.exterior.coords)]
    )

    holes = [
        (2, gmsh.model.occ.addPlaneSurface([make_loop(r.coords)]))
        for r in poly.interiors
    ]

    gmsh.model.occ.synchronize()

    faces, _ = gmsh.model.occ.cut(
        [(2, outer)],
        holes,
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    if len(faces) > 1:
        faces, _ = gmsh.model.occ.fragment(faces, [])
        gmsh.model.occ.synchronize()

    # ---- extrude to volume ----
    gmsh.model.occ.extrude(faces, 0.0, 0.0, z_thickness)
    gmsh.model.occ.synchronize()

    # ---- export STEP (solid B-rep) ----
    for dim in (2, 1, 0):
        gmsh.model.occ.remove(
            gmsh.model.getEntities(dim),
            recursive=True,
        )

    gmsh.model.occ.synchronize()
    gmsh.write(step_path)

    gmsh.finalize()

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


def generate_random_voronoi_network(
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


def lattice_params_and_polygons(Lx, Ly, p, phi_target, center=True, clip=True):
    """
    Finite-size explicit model (valid up to saturation W>=p):
      - Chooses Nx,Ny to fit inside Lx,Ly with square pitch p (allows leftover margins)
      - Computes W from phi_target using the finite-size strip-union model
      - Output polygons (bars) using your edge_rectangles so you can plot

    Returns:
      polys : list[shapely.geometry.Polygon]
      meta  : dict with Nx, Ny, p, W, phi_max, offsets, achieved_phi_model
    """
    # --- choose Nx, Ny to fit inside rectangle ---
    Nx = int(np.floor(Lx / p)) + 1
    Ny = int(np.floor(Ly / p)) + 1
    if Nx < 2 or Ny < 2:
        raise ValueError("Pitch too large: need Nx,Ny >= 2.")

    Lx_lat = (Nx - 1) * p
    Ly_lat = (Ny - 1) * p

    ox = 0.5 * (Lx - Lx_lat) if center else 0.0
    oy = 0.5 * (Ly - Ly_lat) if center else 0.0

    # --- finite-size explicit phi(W) mapping + analytic inversion for W ---
    phi_max = (Lx_lat * Ly_lat) / (Lx * Ly)

    if phi_target >= phi_max:
        W = float(p)  # saturation begins at W = p
    else:
        Astar = phi_target * (Lx * Ly)
        B = (Nx - 1) * Ly_lat + (Ny - 1) * Lx_lat
        C = (Nx - 1) * (Ny - 1)
        disc = B * B - 4.0 * C * Astar
        if disc < 0:
            raise ValueError("Target phi unattainable for this (p,Nx,Ny) in W<p regime.")
        W = float((B - np.sqrt(disc)) / (2.0 * C))

    # model-achieved phi (using the same finite-size mapping)
    if W < p:
        lx = (Nx - 1) * W
        ly = (Ny - 1) * W
    else:
        lx = Lx_lat
        ly = Ly_lat
    A_model = lx * Ly_lat + ly * Lx_lat - lx * ly
    phi_model = A_model / (Lx * Ly)

    # --- build bar polygons using edge_rectangles ---
    x = ox + np.arange(Nx) * p
    y = oy + np.arange(Ny) * p
    xx, yy = np.meshgrid(x, y, indexing="ij")

    h0 = np.stack([xx[:-1, :].ravel(), yy[:-1, :].ravel()], axis=1)
    h1 = np.stack([xx[1:,  :].ravel(), yy[1:,  :].ravel()], axis=1)
    v0 = np.stack([xx[:, :-1].ravel(), yy[:, :-1].ravel()], axis=1)
    v1 = np.stack([xx[:,  1:].ravel(), yy[:,  1:].ravel()], axis=1)

    polys = edge_rectangles(h0, h1, W) + edge_rectangles(v0, v1, W)

    if clip:
        domain = box(0.0, 0.0, Lx, Ly)
        polys = [g.intersection(domain) for g in polys if not g.is_empty]
        polys = [g for g in polys if (not g.is_empty and g.area > 0)]

    meta = {
        "Nx": Nx, "Ny": Ny,
        "p": float(p),
        "W": float(W),
        "phi_target": float(phi_target),
        "phi_model": float(phi_model),
        "phi_max": float(phi_max),
        "Lx_lat": float(Lx_lat), "Ly_lat": float(Ly_lat),
        "ox": float(ox), "oy": float(oy),
    }
    return polys, meta


def plot_polys_black(polys, Lx, Ly, ax=None, figsize=(14, 2), lw=0.2):
    """
    Plot Shapely polygons filled in solid black, with black edges,
    together with the outer rectangle [0,Lx]x[0,Ly].
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Outer rectangle
    # ax.add_patch(Rectangle((0.0, 0.0), Lx, Ly,
                        #    fill=False, edgecolor="black", linewidth=1.5))

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

    for p in polys:
        add_geom(p)

    if len(verts) > 0:
        pc = PolyCollection(
            verts,
            closed=True,
            facecolors="black",
            edgecolors="black",
            linewidths=lw
        )
        ax.add_collection(pc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return ax


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

def generate_polys_for_pattern(pattern: str, Lx: float, Ly: float, phi: float):
    """
    Returns (polys, meta) for a given pattern name.
    Extend this with more patterns as needed.
    """
    if pattern == "square":
        p = Ly / 1
        polys, meta = lattice_params_and_polygons(
            Lx, Ly, p, phi_target=phi, center=True, clip=True
        )
        return polys, meta

    elif pattern == "voronoi":
        n_points = 202
        polys, meta, points, vor = generate_random_voronoi_network(
            Lx=Lx, Ly=Ly, n_points=n_points, phi_target=phi, seed=5, tol=1e-4
        )
        # If you want points/vor later, you can return them too; for now keep same signature.
        return polys, meta

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

if __name__ == "__main__":
    patterns = ["square", "voronoi"]  # add more names here
    Lx = 4.27
    Ly = 1.0
    Lz = 0.16
    phi = 0.15 #areal fraction coverage

    for pattern in patterns:
        polys, meta = generate_polys_for_pattern(pattern, Lx=Lx, Ly=Ly, phi=phi)
        print(f"[{pattern}] Generated {len(polys)} bar polygons with info: {meta}")

        density = area_fraction(polys, Lx, Ly)
        print(f"[{pattern}] Achieved area fraction (density): {density:.6f}")

        # Plot each pattern 
        plot_polys_black(polys, Lx, Ly)
        plt.title(f"{pattern} | φ={density:.4f}")

        # Export to STEP
        step_filename = f"{pattern}_pattern.step"
        export_polygons_to_step(
            polygons = polys,
            z_thickness= Lz,
            step_path = step_filename,
        )
    plt.show()
