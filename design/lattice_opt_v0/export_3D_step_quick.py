"""
3D STEP exporter for optimizer lattices
---------------------------------------

Reads lattice geometry from optimizer HDF5 files and exports a STEP model.

Supported lattice types
- square
- graded hex
- two-skin graded hex
- any future topology stored as nodes/edges

No GUI. User settings are defined below.
"""

import numpy as np
import h5py
import gmsh
import math
import os

# ============================================================
# USER SETTINGS
# ============================================================

H5_FILE = r"results\lattice_opt_graded_hex_20260311_184240\responsegt\data\lattice_opt_responsegt_graded_hex.h5"
ITERATION = "final"          # "initial", "final", or integer like 12
OUTPUT_STEP = r"graded_hex_solid_resgt.step"    # output STEP file path

EXPORT_MODE = "solid"  # "beam" or "solid"


# Solid export parameters
RADIUS_MODE = "area"   # "area" or "constant"

RADIUS_SCALE = 1.0     # used when radius_mode="area"

CONSTANT_RADIUS = 0.003

MIN_RADIUS = 2.5e-4

FUSE_SOLIDS = False


# ============================================================
# helper functions
# ============================================================

def get_iteration_key(h5, iteration):

    keys = sorted([k for k in h5.keys() if k.startswith("iter_")])

    if iteration == "initial":
        return keys[0]

    if iteration == "final":
        return keys[-1]

    if isinstance(iteration, int) or iteration.isdigit():
        return f"iter_{int(iteration):04d}"

    raise ValueError("Invalid iteration value")


def read_lattice(h5file, iteration):

    with h5py.File(h5file, "r") as h5:

        key = get_iteration_key(h5, iteration)

        grp = h5[key]

        nodes = grp["nodes"][()]

        edges = grp["edges"][()].astype(int)

        if "areas" in grp:
            areas = grp["areas"][()]
        else:
            areas = np.ones(edges.shape[0])

    return nodes, edges, areas


def radius_from_area(area):

    if area <= 0:
        return MIN_RADIUS

    r = math.sqrt(area / math.pi)

    return max(RADIUS_SCALE * r, MIN_RADIUS)


# ============================================================
# beam export
# ============================================================

def export_beam(nodes, edges):

    gmsh.initialize()

    gmsh.model.add("lattice")

    occ = gmsh.model.occ

    points = []

    for p in nodes:

        tag = occ.addPoint(p[0], p[1], p[2])

        points.append(tag)

    for i, j in edges:

        occ.addLine(points[i], points[j])

    occ.synchronize()

    gmsh.write(OUTPUT_STEP)

    gmsh.finalize()


# ============================================================
# solid export
# ============================================================

def export_solids(nodes, edges, areas):

    gmsh.initialize()

    gmsh.model.add("lattice")

    occ = gmsh.model.occ

    cylinders = []

    for e, (i, j) in enumerate(edges):

        p = nodes[i]

        q = nodes[j]

        vec = q - p

        L = np.linalg.norm(vec)

        if L == 0:
            continue

        if RADIUS_MODE == "area":

            r = radius_from_area(areas[e])

        else:

            r = max(CONSTANT_RADIUS, MIN_RADIUS)

        tag = occ.addCylinder(
            p[0], p[1], p[2],
            vec[0], vec[1], vec[2],
            r
        )

        cylinders.append((3, tag))

    if FUSE_SOLIDS and len(cylinders) > 1:

        occ.fuse([cylinders[0]], cylinders[1:])

    occ.synchronize()

    gmsh.write(OUTPUT_STEP)

    gmsh.finalize()


# ============================================================
# main
# ============================================================

def main():

    if H5_FILE == "":
        raise RuntimeError("Set H5_FILE at top of script")

    if OUTPUT_STEP == "":
        raise RuntimeError("Set OUTPUT_STEP at top of script")

    nodes, edges, areas = read_lattice(H5_FILE, ITERATION)

    print("Nodes:", nodes.shape[0])
    print("Edges:", edges.shape[0])

    if EXPORT_MODE == "beam":

        export_beam(nodes, edges)

    elif EXPORT_MODE == "solid":

        export_solids(nodes, edges, areas)

    else:

        raise RuntimeError("EXPORT_MODE must be beam or solid")

    print("STEP export complete")
    print(OUTPUT_STEP)


if __name__ == "__main__":
    main()