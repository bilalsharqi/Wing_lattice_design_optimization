"""
Robust CAD STEP exporter for optimizer HDF5 lattices.

This replaces the earlier polygon-buffer CAD exporter.
It preserves the actual truss-member geometry instead of flattening members
into XY polygons and extruding artificial plates.

Recommended use
---------------
- Use export_3D_step_quick.py for very fast beam / solid checks.
- Use this file when you want a more CAD-faithful truss solid export.

Strategy
--------
1) Read nodes / edges / areas from optimizer HDF5.
2) Classify members into top / bottom / mixed-z families.
3) Build true 3D members:
      - beam mode  -> line edges
      - solid mode -> cylinders along each member
4) Optionally add node spheres to improve visual continuity / fusion.
5) Fuse either:
      - none       -> every member stays separate in one STEP compound
      - family     -> fuse top / bottom / mixed families separately
      - global     -> fuse everything together (slowest / least robust)

Notes
-----
- This exporter is geometry-faithful to the graph.
- It does NOT invent planar plates.
- Global boolean fusion of hundreds of members can still be slow / fragile.
  In practice, FUSE_MODE='family' is a better compromise.
"""

import math
import os
import h5py
import numpy as np

try:
    import cadquery as cq
    from cadquery import exporters
except Exception as err:
    raise RuntimeError(
        "cadquery is required for export_3D_step_CAD.py. "
        f"Import failed with: {err}"
    )


# ============================================================
# USER SETTINGS
# ============================================================

H5_FILE = r"results\lattice_opt_graded_hex_20260310_191457\responsegt\data\lattice_opt_responsegt_graded_hex.h5"
ITERATION = "final"          # "initial", "final", or integer like 12
OUTPUT_STEP = r"square_solid.step"

EXPORT_MODE = "solid"        # "beam" or "solid"

# Radius / width control for solid members
RADIUS_MODE = "area"         # "area" or "constant"
RADIUS_SCALE = 1.0            # used if RADIUS_MODE == "area"
CONSTANT_RADIUS = 0.003       # used if RADIUS_MODE == "constant"
MIN_RADIUS = 2.5e-4

# Fusion mode
#   "none"   -> no boolean fuse, just export one compound containing members
#   "family" -> fuse top, bottom, mixed-z families separately
#   "global" -> fuse everything together
FUSE_MODE = "family"

# Optional node joints (solid mode only)
ADD_NODE_SPHERES = True
NODE_SPHERE_MODE = "radius"   # "radius" or "factor"
NODE_SPHERE_RADIUS = 0.0035
NODE_SPHERE_FACTOR = 1.10      # sphere radius = factor * max incident member radius
MIN_NODE_RADIUS = 5e-4

# Fusion batching (used for family/global fusion to improve robustness)
FUSE_BATCH_SIZE = 40

# Which families to include
INCLUDE_TOP = True
INCLUDE_BOTTOM = True
INCLUDE_MIXED = True

# z classification tolerance (None -> auto)
Z_TOL = None

VERBOSE = True


# ============================================================
# HDF5 utilities
# ============================================================

def _resolve_iter_key(h5, iteration):
    keys = sorted([k for k in h5.keys() if k.startswith("iter_")])
    if not keys:
        raise ValueError("No iteration groups found in HDF5")

    if isinstance(iteration, str):
        val = iteration.strip().lower()
        if val == "initial":
            return keys[0]
        if val == "final":
            return keys[-1]
        if val.isdigit():
            key = f"iter_{int(val):04d}"
            if key not in h5:
                raise ValueError(f"Iteration {key} not found in HDF5")
            return key
        raise ValueError("ITERATION must be 'initial', 'final', or an integer")

    key = f"iter_{int(iteration):04d}"
    if key not in h5:
        raise ValueError(f"Iteration {key} not found in HDF5")
    return key


def read_lattice(h5_path, iteration="final"):
    with h5py.File(h5_path, "r") as h5:
        key = _resolve_iter_key(h5, iteration)
        grp = h5[key]
        nodes = grp["nodes"][()]
        edges = grp["edges"][()].astype(int)
        if "areas" in grp:
            areas = grp["areas"][()]
        else:
            areas = np.ones(edges.shape[0], dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"Expected nodes shape (N,3), got {nodes.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edges shape (M,2), got {edges.shape}")
    return nodes, edges, areas, key


# ============================================================
# Geometry helpers
# ============================================================

def radius_from_area(area, radius_scale=1.0, min_radius=1e-4):
    area = float(area)
    if (not np.isfinite(area)) or area <= 0.0:
        return float(min_radius)
    return float(max(radius_scale * math.sqrt(area / math.pi), min_radius))


def member_radius(area):
    if RADIUS_MODE == "constant":
        return max(float(CONSTANT_RADIUS), float(MIN_RADIUS))
    return radius_from_area(area, radius_scale=RADIUS_SCALE, min_radius=MIN_RADIUS)


def classify_edges_by_z(nodes, edges, z_tol=None):
    z = nodes[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    zmid = 0.5 * (zmin + zmax)
    depth = max(zmax - zmin, 1e-12)

    if z_tol is None:
        z_tol = 1e-6 + 1e-3 * depth

    top_mask = np.zeros(len(edges), dtype=bool)
    bot_mask = np.zeros(len(edges), dtype=bool)
    mix_mask = np.zeros(len(edges), dtype=bool)

    for e_idx, (i, j) in enumerate(edges):
        zi = float(nodes[int(i), 2])
        zj = float(nodes[int(j), 2])

        both_top = (zi > zmid + z_tol) and (zj > zmid + z_tol)
        both_bot = (zi < zmid - z_tol) and (zj < zmid - z_tol)

        if both_top:
            top_mask[e_idx] = True
        elif both_bot:
            bot_mask[e_idx] = True
        else:
            mix_mask[e_idx] = True

    return {
        "zmin": zmin,
        "zmax": zmax,
        "zmid": zmid,
        "depth": depth,
        "z_tol": z_tol,
        "top_mask": top_mask,
        "bot_mask": bot_mask,
        "mix_mask": mix_mask,
    }


def _make_beam_edge(p, q):
    return cq.Edge.makeLine(cq.Vector(*p), cq.Vector(*q))


def _make_solid_member(p, q, radius):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    v = q - p
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        return None
    direction = cq.Vector(*(v / L))
    base = cq.Vector(*p)
    return cq.Solid.makeCylinder(float(radius), L, base, direction)


def _make_node_sphere(center, radius):
    return cq.Solid.makeSphere(float(radius), cq.Vector(*center))


def _compound_from_shapes(shapes):
    shapes = [s for s in shapes if s is not None]
    if len(shapes) == 0:
        return None
    return cq.Compound.makeCompound(shapes)


def _fuse_list_batched(shapes, batch_size=40):
    """
    Batched boolean fusion to improve robustness relative to a single giant fuse.
    Returns a single shape or None.
    """
    shapes = [s for s in shapes if s is not None]
    if len(shapes) == 0:
        return None
    if len(shapes) == 1:
        return shapes[0]

    current = shapes[:]
    while len(current) > 1:
        next_round = []
        for start in range(0, len(current), batch_size):
            batch = current[start:start + batch_size]
            base = batch[0]
            for s in batch[1:]:
                try:
                    base = base.fuse(s)
                except Exception:
                    # fallback: keep in compound if fuse fails locally
                    base = cq.Compound.makeCompound([base, s])
            next_round.append(base)
        current = next_round

    return current[0]


def _family_edge_indices(zinfo):
    fam = {}
    fam["top"] = np.where(zinfo["top_mask"])[0]
    fam["bottom"] = np.where(zinfo["bot_mask"])[0]
    fam["mixed"] = np.where(zinfo["mix_mask"])[0]
    return fam


def _node_family_membership(nodes, edges, family_indices, areas):
    """
    Return per-node max radius based on incident edges in the selected families.
    Used for optional node spheres.
    """
    node_r = np.zeros(nodes.shape[0], dtype=float)
    for fam_name, idxs in family_indices.items():
        for e_idx in idxs:
            i, j = edges[e_idx]
            r = member_radius(areas[e_idx])
            node_r[int(i)] = max(node_r[int(i)], r)
            node_r[int(j)] = max(node_r[int(j)], r)
    return node_r


def _selected_family_indices(zinfo):
    fam = _family_edge_indices(zinfo)
    selected = {}
    if INCLUDE_TOP:
        selected["top"] = fam["top"]
    if INCLUDE_BOTTOM:
        selected["bottom"] = fam["bottom"]
    if INCLUDE_MIXED:
        selected["mixed"] = fam["mixed"]
    return selected


# ============================================================
# Export builder
# ============================================================

def build_export_shape(nodes, edges, areas, zinfo):
    selected = _selected_family_indices(zinfo)
    if len(selected) == 0:
        raise ValueError("No edge families selected for export")

    if EXPORT_MODE not in ("beam", "solid"):
        raise ValueError("EXPORT_MODE must be 'beam' or 'solid'")
    if FUSE_MODE not in ("none", "family", "global"):
        raise ValueError("FUSE_MODE must be 'none', 'family', or 'global'")

    family_shapes = {}

    if EXPORT_MODE == "beam":
        for fam_name, idxs in selected.items():
            beam_edges = []
            for e_idx in idxs:
                i, j = edges[e_idx]
                beam_edges.append(_make_beam_edge(nodes[int(i)], nodes[int(j)]))
            family_shapes[fam_name] = _compound_from_shapes(beam_edges)

        if FUSE_MODE == "none":
            return _compound_from_shapes(list(family_shapes.values())), family_shapes
        # beam objects cannot be meaningfully boolean-fused like solids; keep as compound
        return _compound_from_shapes(list(family_shapes.values())), family_shapes

    # solid mode
    for fam_name, idxs in selected.items():
        solids = []
        for e_idx in idxs:
            i, j = edges[e_idx]
            r = member_radius(areas[e_idx])
            solids.append(_make_solid_member(nodes[int(i)], nodes[int(j)], r))
        family_shapes[fam_name] = solids

    # optional node spheres
    if ADD_NODE_SPHERES:
        node_r = _node_family_membership(nodes, edges, selected, areas)
        sphere_shapes = []
        for n_idx, r_edge in enumerate(node_r):
            if r_edge <= 0.0:
                continue
            if NODE_SPHERE_MODE == "radius":
                r = max(float(NODE_SPHERE_RADIUS), float(MIN_NODE_RADIUS))
            else:
                r = max(float(NODE_SPHERE_FACTOR) * float(r_edge), float(MIN_NODE_RADIUS))
            sphere_shapes.append(_make_node_sphere(nodes[n_idx], r))

        # attach spheres to mixed family if present, else top, else bottom
        if "mixed" in family_shapes:
            family_shapes["mixed"].extend(sphere_shapes)
        elif "top" in family_shapes:
            family_shapes["top"].extend(sphere_shapes)
        elif "bottom" in family_shapes:
            family_shapes["bottom"].extend(sphere_shapes)

    fused_families = {}

    if FUSE_MODE == "none":
        all_shapes = []
        for fam_name, solids in family_shapes.items():
            fused_families[fam_name] = _compound_from_shapes(solids)
            if fused_families[fam_name] is not None:
                all_shapes.append(fused_families[fam_name])
        return _compound_from_shapes(all_shapes), fused_families

    if FUSE_MODE == "family":
        all_shapes = []
        for fam_name, solids in family_shapes.items():
            fused_families[fam_name] = _fuse_list_batched(solids, batch_size=FUSE_BATCH_SIZE)
            if fused_families[fam_name] is not None:
                all_shapes.append(fused_families[fam_name])
        # keep families as separate bodies inside one STEP compound
        return _compound_from_shapes(all_shapes), fused_families

    # global
    merged = []
    for fam_name, solids in family_shapes.items():
        fam_shape = _fuse_list_batched(solids, batch_size=FUSE_BATCH_SIZE)
        fused_families[fam_name] = fam_shape
        if fam_shape is not None:
            merged.append(fam_shape)
    global_shape = _fuse_list_batched(merged, batch_size=max(2, FUSE_BATCH_SIZE // 2))
    return global_shape, fused_families


# ============================================================
# Main export function
# ============================================================

def export_lattice_step_cad(h5_path, output_step, iteration="final"):
    nodes, edges, areas, iter_key = read_lattice(h5_path, iteration=iteration)
    zinfo = classify_edges_by_z(nodes, edges, z_tol=Z_TOL)
    export_shape, family_shapes = build_export_shape(nodes, edges, areas, zinfo)

    if export_shape is None:
        raise ValueError("No geometry was created for export")

    os.makedirs(os.path.dirname(os.path.abspath(output_step)), exist_ok=True)
    exporters.export(export_shape, output_step)

    fam = _family_edge_indices(zinfo)
    return {
        "output_step": output_step,
        "iteration_key": iter_key,
        "n_nodes": int(nodes.shape[0]),
        "n_edges": int(edges.shape[0]),
        "zmin": float(zinfo["zmin"]),
        "zmax": float(zinfo["zmax"]),
        "depth": float(zinfo["depth"]),
        "n_top_edges": int(len(fam["top"])),
        "n_bottom_edges": int(len(fam["bottom"])),
        "n_mixed_edges": int(len(fam["mixed"])),
        "export_mode": EXPORT_MODE,
        "fuse_mode": FUSE_MODE,
        "add_node_spheres": bool(ADD_NODE_SPHERES and EXPORT_MODE == "solid"),
    }


# ============================================================
# Script entry point
# ============================================================

def main():
    if not H5_FILE:
        raise RuntimeError("Set H5_FILE at top of script")
    if not OUTPUT_STEP:
        raise RuntimeError("Set OUTPUT_STEP at top of script")

    result = export_lattice_step_cad(
        h5_path=H5_FILE,
        output_step=OUTPUT_STEP,
        iteration=ITERATION,
    )

    if VERBOSE:
        print("CAD STEP export complete.")
        print(f"File: {os.path.abspath(result['output_step'])}")
        print(f"Iteration: {result['iteration_key']}")
        print(f"Nodes: {result['n_nodes']}")
        print(f"Edges: {result['n_edges']}")
        print(f"z range: [{result['zmin']:.6f}, {result['zmax']:.6f}]")
        print(f"Depth: {result['depth']:.6f}")
        print(f"Top edges: {result['n_top_edges']}")
        print(f"Bottom edges: {result['n_bottom_edges']}")
        print(f"Mixed edges: {result['n_mixed_edges']}")
        print(f"Export mode: {result['export_mode']}")
        print(f"Fuse mode: {result['fuse_mode']}")
        print(f"Node spheres: {result['add_node_spheres']}")


if __name__ == "__main__":
    main()
