# generate_two_skin_graded_hex_wingbox_lattice.py

import numpy as np
from dataclasses import dataclass


@dataclass
class Lattice:
    nodes: np.ndarray
    edges: np.ndarray


def _graded_span_positions(span, ny, root_scale, tip_scale):
    """
    True spanwise grading:
    smaller bays near root if root_scale < tip_scale.
    Returns ny+1 spanwise node rows from y=0 to y=span.
    """
    ny = int(max(2, ny))
    t = np.linspace(0.0, 1.0, ny)
    weights = (1.0 - t) * float(root_scale) + t * float(tip_scale)
    s = np.concatenate(([0.0], np.cumsum(weights)))
    s /= s[-1]
    return float(span) * s


def _row_x_positions_rectangular_perimeter(chord, nx, row_index):
    """
    Hex-like staggered row with a hard rectangular perimeter.

    Boundary nodes are always kept at:
        x = 0
        x = chord

    Only interior nodes are staggered.
    """
    nx = int(max(2, nx))
    base = np.linspace(0.0, float(chord), nx + 1)

    if len(base) <= 2:
        return base

    interior = base[1:-1].copy()

    if row_index % 2 == 1:
        dx = base[1] - base[0]
        interior = interior + 0.5 * dx

    interior = np.clip(interior, 0.0, float(chord))

    xs = np.concatenate(([0.0], interior, [float(chord)]))
    xs = np.unique(np.round(xs, 12))
    return xs


def generate_two_skin_graded_hex_wingbox_lattice(
    span,
    chord,
    depth,
    ny=14,
    nx=6,
    root_cell_scale=0.65,
    tip_cell_scale=1.35,
    add_skin_diagonals=True,
    add_verticals=True,
    add_yz_diagonals=True,
    add_xz_diagonals=True,
    add_perimeter_frame=True,
):
    """
    Two-skin graded hex-like / triangular wingbox lattice.

    Compatible with the configured optimizer call signature.

    Axes:
        x = chordwise
        y = spanwise
        z = vertical / thickness

    Features:
    - true spanwise grading via row spacing
    - top and bottom skins
    - optional verticals and skin-to-skin diagonals
    - hard rectangular perimeter
    """

    ny = int(max(2, ny))
    nx = int(max(2, nx))

    y_rows = _graded_span_positions(
        span=span,
        ny=ny,
        root_scale=root_cell_scale,
        tip_scale=tip_cell_scale,
    )

    # ------------------------------------------------------------
    # Build top skin nodes
    # ------------------------------------------------------------
    top_nodes = []
    row_ids = []

    for j, y in enumerate(y_rows):
        xs = _row_x_positions_rectangular_perimeter(chord, nx, j)
        ids = []
        for x in xs:
            ids.append(len(top_nodes))
            top_nodes.append([float(x), float(y), 0.5 * float(depth)])
        row_ids.append(ids)

    top_nodes = np.asarray(top_nodes, dtype=float)
    n_top = len(top_nodes)

    bottom_nodes = top_nodes.copy()
    bottom_nodes[:, 2] = -0.5 * float(depth)

    nodes = np.vstack((top_nodes, bottom_nodes))

    # ------------------------------------------------------------
    # Edge helper
    # ------------------------------------------------------------
    edge_set = set()

    def add_edge(i, j):
        i = int(i)
        j = int(j)
        if i == j:
            return
        a, b = sorted((i, j))
        edge_set.add((a, b))

    # ------------------------------------------------------------
    # Skin connectivity
    # ------------------------------------------------------------
    def connect_skin(offset):
        # chordwise members
        for ids in row_ids:
            for a, b in zip(ids[:-1], ids[1:]):
                add_edge(offset + a, offset + b)

        # spanwise nearest-neighbor stitching
        for j in range(len(row_ids) - 1):
            r0 = row_ids[j]
            r1 = row_ids[j + 1]

            x0 = np.array([top_nodes[i, 0] for i in r0])
            x1 = np.array([top_nodes[i, 0] for i in r1])

            map01 = [int(np.argmin(np.abs(x1 - xx))) for xx in x0]
            map10 = [int(np.argmin(np.abs(x0 - xx))) for xx in x1]

            for k, a in enumerate(r0):
                kk = map01[k]
                add_edge(offset + a, offset + r1[kk])

                if add_skin_diagonals:
                    if kk > 0:
                        add_edge(offset + a, offset + r1[kk - 1])
                    if kk < len(r1) - 1:
                        add_edge(offset + a, offset + r1[kk + 1])

            # reverse pass for symmetry / robustness
            for k, b in enumerate(r1):
                kk = map10[k]
                add_edge(offset + b, offset + r0[kk])

    connect_skin(0)
    connect_skin(n_top)

    # ------------------------------------------------------------
    # Through-thickness members
    # ------------------------------------------------------------
    if add_verticals:
        for i in range(n_top):
            add_edge(i, i + n_top)

    if add_yz_diagonals:
        for j in range(len(row_ids) - 1):
            r0 = row_ids[j]
            r1 = row_ids[j + 1]
            m = min(len(r0), len(r1))
            for k in range(m):
                a = r0[k]
                b = r1[k]
                add_edge(a, b + n_top)
                add_edge(a + n_top, b)

    if add_xz_diagonals:
        for ids in row_ids:
            for a, b in zip(ids[:-1], ids[1:]):
                add_edge(a, b + n_top)
                add_edge(a + n_top, b)

    # ------------------------------------------------------------
    # Explicit rectangular perimeter frame
    # ------------------------------------------------------------
    if add_perimeter_frame:
        for offset in (0, n_top):
            root_ids = row_ids[0]
            tip_ids = row_ids[-1]

            # root and tip row edges
            for a, b in zip(root_ids[:-1], root_ids[1:]):
                add_edge(offset + a, offset + b)
            for a, b in zip(tip_ids[:-1], tip_ids[1:]):
                add_edge(offset + a, offset + b)

            # leading and trailing edge chains
            for j in range(len(row_ids) - 1):
                r0 = row_ids[j]
                r1 = row_ids[j + 1]
                add_edge(offset + r0[0], offset + r1[0])     # LE
                add_edge(offset + r0[-1], offset + r1[-1])   # TE

        # corner posts
        top_root = row_ids[0]
        top_tip = row_ids[-1]
        for idx in (top_root[0], top_root[-1], top_tip[0], top_tip[-1]):
            add_edge(idx, idx + n_top)

    edges = np.asarray(sorted(edge_set), dtype=int)
    return Lattice(nodes=nodes, edges=edges)


def generate_two_skin_uniform_hex_wingbox_lattice(
    span,
    chord,
    depth,
    ny=14,
    nx=6,
    add_skin_diagonals=True,
    add_verticals=True,
    add_yz_diagonals=True,
    add_xz_diagonals=True,
    add_perimeter_frame=True,
):
    return generate_two_skin_graded_hex_wingbox_lattice(
        span=span,
        chord=chord,
        depth=depth,
        ny=ny,
        nx=nx,
        root_cell_scale=1.0,
        tip_cell_scale=1.0,
        add_skin_diagonals=add_skin_diagonals,
        add_verticals=add_verticals,
        add_yz_diagonals=add_yz_diagonals,
        add_xz_diagonals=add_xz_diagonals,
        add_perimeter_frame=add_perimeter_frame,
    )


if __name__ == "__main__":
    lat = generate_two_skin_graded_hex_wingbox_lattice(
        span=4.0,
        chord=1.0,
        depth=0.16,
        ny=14,
        nx=6,
        root_cell_scale=0.50,
        tip_cell_scale=1.80,
        add_skin_diagonals=True,
        add_verticals=True,
        add_yz_diagonals=True,
        add_xz_diagonals=True,
        add_perimeter_frame=True,
    )
    print("nodes:", lat.nodes.shape)
    print("edges:", lat.edges.shape)
    print("x-range:", lat.nodes[:, 0].min(), lat.nodes[:, 0].max())
    print("y-range:", lat.nodes[:, 1].min(), lat.nodes[:, 1].max())
    print("z-range:", lat.nodes[:, 2].min(), lat.nodes[:, 2].max())