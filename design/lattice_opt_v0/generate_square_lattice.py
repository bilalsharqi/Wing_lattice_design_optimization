import numpy as np
from dataclasses import dataclass


@dataclass
class Lattice:
    nodes: np.ndarray
    edges: np.ndarray


def _graded_span_positions(span, ny, root_cell_scale=1.0, tip_cell_scale=1.0):
    """
    True spanwise grading:
    smaller bays near root if root_cell_scale < tip_cell_scale.
    Returns ny positions from 0 to span.
    """
    ny = int(max(2, ny))
    t = np.linspace(0.0, 1.0, ny)
    weights = (1.0 - t) * float(root_cell_scale) + t * float(tip_cell_scale)
    s = np.concatenate(([0.0], np.cumsum(weights)))
    s /= s[-1]
    ys = float(span) * s
    return ys[:-1] if len(ys) > ny else ys


def generate_square_wingbox_lattice(
    span,
    chord,
    depth,
    ny=21,
    nx=7,
    nz=3,
    root_cell_scale=1.0,
    tip_cell_scale=1.0,
    add_xy_diagonals=True,
    add_yz_diagonals=False,
    add_xz_diagonals=False,
):
    """
    Generate a square wingbox lattice.

    Axes:
      x = chordwise
      y = spanwise
      z = thickness / vertical

    Backward compatible:
      - if root_cell_scale = tip_cell_scale = 1.0, this reduces to the old uniform grid
      - existing optimizer calls still work
    """
    nx = int(max(2, nx))
    ny = int(max(2, ny))
    nz = int(max(2, nz))

    xs = np.linspace(0.0, float(chord), nx)
    ys = _graded_span_positions(float(span), ny, root_cell_scale, tip_cell_scale)
    zs = np.linspace(-float(depth) / 2.0, float(depth) / 2.0, nz)

    nodes = []
    node_id = {}

    idx = 0
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                nodes.append([x, y, z])
                node_id[(i, j, k)] = idx
                idx += 1

    nodes = np.asarray(nodes, dtype=float)

    edge_set = set()

    def add_edge(a, b):
        if a == b:
            return
        i, j = sorted((int(a), int(b)))
        edge_set.add((i, j))

    # Orthogonal grid members
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = node_id[(i, j, k)]

                if i + 1 < nx:
                    add_edge(n0, node_id[(i + 1, j, k)])  # x-direction

                if j + 1 < ny:
                    add_edge(n0, node_id[(i, j + 1, k)])  # y-direction

                if k + 1 < nz:
                    add_edge(n0, node_id[(i, j, k + 1)])  # z-direction

    # XY diagonals on each z-layer
    if add_xy_diagonals:
        for k in range(nz):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    n00 = node_id[(i, j, k)]
                    n10 = node_id[(i + 1, j, k)]
                    n01 = node_id[(i, j + 1, k)]
                    n11 = node_id[(i + 1, j + 1, k)]

                    if (i + j + k) % 2 == 0:
                        add_edge(n00, n11)
                    else:
                        add_edge(n10, n01)

    # YZ diagonals on each x-plane
    if add_yz_diagonals:
        for i in range(nx):
            for k in range(nz - 1):
                for j in range(ny - 1):
                    n00 = node_id[(i, j, k)]
                    n10 = node_id[(i, j + 1, k)]
                    n01 = node_id[(i, j, k + 1)]
                    n11 = node_id[(i, j + 1, k + 1)]

                    if (i + j + k) % 2 == 0:
                        add_edge(n00, n11)
                    else:
                        add_edge(n10, n01)

    # XZ diagonals on each y-plane
    if add_xz_diagonals:
        for j in range(ny):
            for k in range(nz - 1):
                for i in range(nx - 1):
                    n00 = node_id[(i, j, k)]
                    n10 = node_id[(i + 1, j, k)]
                    n01 = node_id[(i, j, k + 1)]
                    n11 = node_id[(i + 1, j, k + 1)]

                    if (i + j + k) % 2 == 0:
                        add_edge(n00, n11)
                    else:
                        add_edge(n10, n01)

    edges = np.asarray(sorted(edge_set), dtype=int)
    return Lattice(nodes=nodes, edges=edges)


if __name__ == "__main__":
    lat = generate_square_wingbox_lattice(
        span=4.0,
        chord=1.0,
        depth=0.16,
        ny=14,
        nx=6,
        nz=2,
        root_cell_scale=0.6,
        tip_cell_scale=1.5,
        add_xy_diagonals=True,
        add_yz_diagonals=False,
        add_xz_diagonals=False,
    )
    print("nodes:", lat.nodes.shape)
    print("edges:", lat.edges.shape)
    print("y min/max:", lat.nodes[:, 1].min(), lat.nodes[:, 1].max())