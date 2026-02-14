from __future__ import annotations
from typing import Iterable, Tuple, Dict, List, Sequence
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
# ---------- stacked exporter (N layers) ----------

def export_stacked_polygons_to_step(
    layers: Sequence[Tuple[Iterable[Polygon], float]],
    step_path: str,
    *,
    model_name: str = "stacked_polys",
    tol_snap: float = 1e-6,
    simplify_tol: float = 0.0,
    per_poly_buffer0: bool = True,
    occ_fix: bool = True,
) -> None:
    """
    Export an arbitrary number of stacked polygon layers to a *single* STEP file.

    Parameters
    ----------
    layers:
        Sequence of (polygons, z_thickness) tuples.
        Ordering matters: layer[0] is the bottom. Each next layer is placed on top
        of the previous one (its z0 is cumulative sum of prior thicknesses).
    """

    if not layers:
        raise ValueError("layers is empty")

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

    # ---- vertex handling (reuse your snap+cache) ----
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
        if len(ring) < 3:
            raise ValueError("Degenerate ring after snapping; increase tol_snap or fix polygon.")
        pts = [add_point(x, y) for x, y in ring]
        lines = [
            gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
            for i in range(len(pts))
        ]
        return gmsh.model.occ.addCurveLoop(lines)

    def build_faces_from_geom(geom) -> List[Tuple[int, int]]:
        """
        Convert (Polygon|MultiPolygon|GeometryCollection) into planar faces (dim=2 tags),
        with holes correctly subtracted.
        """
        faces_out: List[Tuple[int, int]] = []

        # NOTE: do NOT call removeAllDuplicates() inside this function.
        # It can delete/merge points and invalidate the add_point cache.

        for p in _iter_polys(geom):
            if p.is_empty or p.area <= 0:
                continue
            p = orient(p, 1.0)

            outer_loop = make_loop(p.exterior.coords)
            outer_surf = gmsh.model.occ.addPlaneSurface([outer_loop])

            hole_tools: List[Tuple[int, int]] = []
            for r in p.interiors:
                hole_loop = make_loop(r.coords)
                hole_surf = gmsh.model.occ.addPlaneSurface([hole_loop])
                hole_tools.append((2, hole_surf))

            gmsh.model.occ.synchronize()

            if hole_tools:
                cut_faces, _ = gmsh.model.occ.cut(
                    [(2, outer_surf)],
                    hole_tools,
                    removeObject=True,
                    removeTool=True,
                )
                gmsh.model.occ.synchronize()
                faces_out.extend(cut_faces)
            else:
                faces_out.append((2, outer_surf))

        # Fragment once at the end if needed
        if len(faces_out) > 1:
            faces_out, _ = gmsh.model.occ.fragment(faces_out, [])
            gmsh.model.occ.synchronize()

        return faces_out
    def extrude_faces_to_volumes(faces: List[Tuple[int, int]], dz: float) -> List[Tuple[int, int]]:
        """
        Extrude given faces by dz; return created volume entities (dim=3).
        """
        if not faces or dz == 0:
            return []
        out = gmsh.model.occ.extrude(faces, 0.0, 0.0, dz)
        gmsh.model.occ.synchronize()
        return [(dim, tag) for (dim, tag) in out if dim == 3]

    # ---- build + extrude each layer, stack by cumulative z ----
    all_layer_vols: List[List[Tuple[int, int]]] = []
    z0 = 0.0

    for i, (polygons, thickness) in enumerate(layers):
        if thickness < 0:
            gmsh.finalize()
            raise ValueError(f"Layer {i} has negative thickness: {thickness}")

        prepped = _preprocess(polygons, buffer0=per_poly_buffer0, simplify=simplify_tol)
        geom = unary_union(prepped) if prepped else GeometryCollection()

        faces = build_faces_from_geom(geom)
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        cache.clear()  # IMPORTANT: point tags may have changed
        vols = extrude_faces_to_volumes(faces, float(thickness))

        if z0 != 0.0 and vols:
            gmsh.model.occ.translate(vols, 0.0, 0.0, z0)
            gmsh.model.occ.synchronize()

        all_layer_vols.append(vols)
        z0 += float(thickness)

    # ---- fuse all layers into one solid (when they overlap/touch by area) ----
    # start from first non-empty layer
    fused: List[Tuple[int, int]] = []
    for vols in all_layer_vols:
        if vols:
            fused = vols
            break

    if not fused:
        gmsh.finalize()
        raise ValueError("All layers are empty after preprocessing/union; nothing to export.")

    for vols in all_layer_vols:
        if not vols or vols is fused:
            continue
        fused, _ = gmsh.model.occ.fuse(fused, vols, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # ---- Cleanup: keep only volumes before writing STEP ----
    for dim in (2, 1, 0):
        gmsh.model.occ.remove(gmsh.model.getEntities(dim), recursive=True)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    print("n_volumes =", len(vols), "vol_tags =", [t for (_, t) in vols])
    gmsh.write(step_path)  # ensure .step or .stp
    gmsh.finalize()

## example usage:
# layers = [
#     (polys0, 1.0),   # bottom, thickness 1.0
#     (polys1, 0.5),   # starts at z=1.0, thickness of layer 0.5
#     (polys2, 2.0),   # starts at z=1.5, thickness of layer 2.0
# ]
# export_stacked_polygons_to_step(layers, r"C:\tmp\stacked.step")