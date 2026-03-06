import numpy as np
from dataclasses import dataclass


@dataclass
class Lattice3D:
    nodes: np.ndarray
    edges: np.ndarray


def generate_octet_ground_structure(span, chord, depth, ny=21, nx=7, nz=5):
    xs = np.linspace(0.0, chord, nx)
    ys = np.linspace(0.0, span, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)

    nodes = np.array([(x, y, z) for y in ys for x in xs for z in zs], dtype=float)
    idx = lambda ix, iy, iz: (iy * nx * nz) + (ix * nz) + iz

    edges_set = set()

    def add_edge(i, j):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        edges_set.add((a, b))

    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                i = idx(ix, iy, iz)

                if ix + 1 < nx:
                    add_edge(i, idx(ix + 1, iy, iz))
                if iy + 1 < ny:
                    add_edge(i, idx(ix, iy + 1, iz))
                if iz + 1 < nz:
                    add_edge(i, idx(ix, iy, iz + 1))

                if ix + 1 < nx and iy + 1 < ny:
                    add_edge(i, idx(ix + 1, iy + 1, iz))
                    add_edge(idx(ix + 1, iy, iz), idx(ix, iy + 1, iz))

                if ix + 1 < nx and iz + 1 < nz:
                    add_edge(i, idx(ix + 1, iy, iz + 1))
                    add_edge(idx(ix + 1, iy, iz), idx(ix, iy, iz + 1))

                if iy + 1 < ny and iz + 1 < nz:
                    add_edge(i, idx(ix, iy + 1, iz + 1))
                    add_edge(idx(ix, iy + 1, iz), idx(ix, iy, iz + 1))

                if ix + 1 < nx and iy + 1 < ny and iz + 1 < nz:
                    add_edge(i, idx(ix + 1, iy + 1, iz + 1))
                    add_edge(idx(ix + 1, iy, iz), idx(ix, iy + 1, iz + 1))
                    add_edge(idx(ix, iy + 1, iz), idx(ix + 1, iy, iz + 1))
                    add_edge(idx(ix, iy, iz + 1), idx(ix + 1, iy + 1, iz))

    edges = np.array(sorted(list(edges_set)), dtype=int)
    return Lattice3D(nodes=nodes, edges=edges)