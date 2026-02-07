from __future__ import annotations
from typing import Iterable, Tuple, Dict, List, Union
import gmsh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


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