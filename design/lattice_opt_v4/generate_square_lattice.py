import numpy as np
from dataclasses import dataclass


@dataclass
class Lattice3D:
    nodes: np.ndarray
    edges: np.ndarray


def generate_square_wingbox_lattice(
    span,
    chord,
    depth,
    ny=21,
    nx=7,
    nz=3,
    add_xy_diagonals=True,
    add_yz_diagonals=False,
    add_xz_diagonals=False,
):
    """
    Generate a simple, interpretable square wing-box lattice.

    Coordinates:
      x = chordwise
      y = spanwise
      z = depth

    Topology:
      - spanwise members: spar-like rails
      - chordwise members: rib-like members
      - vertical members: posts between top/bottom layers
      - optional diagonals on selected planes

    Recommended debug baseline:
      nz=3
      add_xy_diagonals=True
      add_yz_diagonals=False
      add_xz_diagonals=False
    """

    xs = np.linspace(0.0, chord, nx)
    ys = np.linspace(0.0, span, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)

    nodes = np.array([(x, y, z) for y in ys for x in xs for z in zs], dtype=float)

    def idx(ix, iy, iz):
        return iy * nx * nz + ix * nz + iz

    edges_set = set()

    def add_edge(i, j):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        edges_set.add((a, b))

    # -----------------------------
    # Primary square lattice members
    # -----------------------------

    # 1) Spanwise members (spar-like rails)
    for iy in range(ny - 1):
        for ix in range(nx):
            for iz in range(nz):
                add_edge(idx(ix, iy, iz), idx(ix, iy + 1, iz))

    # 2) Chordwise members (rib-like members)
    for iy in range(ny):
        for ix in range(nx - 1):
            for iz in range(nz):
                add_edge(idx(ix, iy, iz), idx(ix + 1, iy, iz))

    # 3) Vertical members (posts)
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz - 1):
                add_edge(idx(ix, iy, iz), idx(ix, iy, iz + 1))

    # -----------------------------
    # Optional diagonals
    # -----------------------------

    # Diagonals in x-y planes (topology of square panels on each z layer)
    if add_xy_diagonals:
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                for iz in range(nz):
                    add_edge(idx(ix, iy, iz), idx(ix + 1, iy + 1, iz))
                    add_edge(idx(ix + 1, iy, iz), idx(ix, iy + 1, iz))

    # Diagonals in y-z planes
    if add_yz_diagonals:
        for iy in range(ny - 1):
            for ix in range(nx):
                for iz in range(nz - 1):
                    add_edge(idx(ix, iy, iz), idx(ix, iy + 1, iz + 1))
                    add_edge(idx(ix, iy + 1, iz), idx(ix, iy, iz + 1))

    # Diagonals in x-z planes
    if add_xz_diagonals:
        for iy in range(ny):
            for ix in range(nx - 1):
                for iz in range(nz - 1):
                    add_edge(idx(ix, iy, iz), idx(ix + 1, iy, iz + 1))
                    add_edge(idx(ix + 1, iy, iz), idx(ix, iy, iz + 1))

    edges = np.array(sorted(edges_set), dtype=int)
    return Lattice3D(nodes=nodes, edges=edges)