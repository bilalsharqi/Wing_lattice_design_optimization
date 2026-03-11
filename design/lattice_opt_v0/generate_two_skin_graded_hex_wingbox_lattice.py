
from dataclasses import dataclass
import numpy as np


@dataclass
class Lattice:
    nodes: np.ndarray
    edges: np.ndarray


def _row_x_positions(chord, nx, frac, root_cell_scale, tip_cell_scale):
    scale = (1.0 - frac) * root_cell_scale + frac * tip_cell_scale
    n_intervals = max(1, int(nx))
    t = np.linspace(0.0, 1.0, n_intervals)
    w = 1.0 + (scale - 1.0) * t
    s = np.concatenate(([0.0], np.cumsum(w)))
    s /= s[-1]
    return chord * s


def generate_two_skin_graded_hex_wingbox_lattice(
    span: float,
    chord: float,
    depth: float,
    ny: int = 14,
    nx: int = 6,
    root_cell_scale: float = 0.65,
    tip_cell_scale: float = 1.35,
    add_skin_diagonals: bool = True,
    add_verticals: bool = True,
    add_yz_diagonals: bool = True,
    add_xz_diagonals: bool = True,
    add_perimeter_frame: bool = True,
):
    ny = int(max(2, ny))
    nx = int(max(2, nx))

    y_rows = np.linspace(0.0, float(span), ny + 1)

    top_nodes = []
    row_ids = []

    for j, y in enumerate(y_rows):
        frac = 0.0 if span <= 0.0 else y / span
        xs = _row_x_positions(
            chord=float(chord),
            nx=nx,
            frac=frac,
            root_cell_scale=float(root_cell_scale),
            tip_cell_scale=float(tip_cell_scale),
        )

        if j % 2 == 1 and len(xs) > 1:
            dx_mean = np.mean(np.diff(xs))
            xs = np.clip(xs + 0.5 * dx_mean, 0.0, chord)

        xs = np.unique(np.round(xs, 12))

        ids_this_row = []
        for x in xs:
            ids_this_row.append(len(top_nodes))
            top_nodes.append([float(x), float(y), 0.5 * float(depth)])
        row_ids.append(ids_this_row)

    top_nodes = np.asarray(top_nodes, dtype=float)
    n_top = top_nodes.shape[0]

    bottom_nodes = top_nodes.copy()
    bottom_nodes[:, 2] = -0.5 * float(depth)

    nodes = np.vstack([top_nodes, bottom_nodes])

    edge_set = set()

    def add_edge(i, j):
        i = int(i)
        j = int(j)
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        edge_set.add((a, b))

    def connect_skin_rows(offset):
        for ids in row_ids:
            for a, b in zip(ids[:-1], ids[1:]):
                add_edge(offset + a, offset + b)

        for j in range(len(row_ids) - 1):
            ra = row_ids[j]
            rb = row_ids[j + 1]

            xa = np.array([top_nodes[i, 0] for i in ra])
            xb = np.array([top_nodes[i, 0] for i in rb])

            for ia, a in enumerate(ra):
                k = int(np.argmin(np.abs(xb - xa[ia])))
                add_edge(offset + a, offset + rb[k])
                if k - 1 >= 0:
                    add_edge(offset + a, offset + rb[k - 1])
                if k + 1 < len(rb):
                    add_edge(offset + a, offset + rb[k + 1])

            if add_skin_diagonals:
                for ib, b in enumerate(rb):
                    k = int(np.argmin(np.abs(xa - xb[ib])))
                    add_edge(offset + b, offset + ra[k])

    connect_skin_rows(offset=0)
    connect_skin_rows(offset=n_top)

    if add_verticals:
        for i in range(n_top):
            add_edge(i, i + n_top)

    if add_yz_diagonals or add_xz_diagonals:
        for j in range(len(row_ids) - 1):
            ra = row_ids[j]
            rb = row_ids[j + 1]

            for idx in range(min(len(ra), len(rb))):
                a = ra[idx]
                b = rb[idx]
                if add_yz_diagonals:
                    add_edge(a, b + n_top)
                    add_edge(a + n_top, b)

            if add_xz_diagonals:
                for ids in (ra, rb):
                    for k in range(len(ids) - 1):
                        i0 = ids[k]
                        i1 = ids[k + 1]
                        add_edge(i0, i1 + n_top)
                        add_edge(i0 + n_top, i1)

    if add_perimeter_frame:
        for offset in (0, n_top):
            root_ids = row_ids[0]
            tip_ids = row_ids[-1]
            for a, b in zip(root_ids[:-1], root_ids[1:]):
                add_edge(offset + a, offset + b)
            for a, b in zip(tip_ids[:-1], tip_ids[1:]):
                add_edge(offset + a, offset + b)

        for offset in (0, n_top):
            for j in range(len(row_ids) - 1):
                ra = row_ids[j]
                rb = row_ids[j + 1]
                add_edge(offset + ra[0], offset + rb[0])
                add_edge(offset + ra[-1], offset + rb[-1])

    edges = np.asarray(sorted(edge_set), dtype=int)
    return Lattice(nodes=nodes, edges=edges)


def generate_two_skin_uniform_hex_wingbox_lattice(
    span: float,
    chord: float,
    depth: float,
    ny: int = 14,
    nx: int = 6,
    **kwargs,
):
    return generate_two_skin_graded_hex_wingbox_lattice(
        span=span,
        chord=chord,
        depth=depth,
        ny=ny,
        nx=nx,
        root_cell_scale=1.0,
        tip_cell_scale=1.0,
        **kwargs,
    )


if __name__ == "__main__":
    lat = generate_two_skin_graded_hex_wingbox_lattice(
        span=4.0,
        chord=1.0,
        depth=0.16,
        ny=14,
        nx=6,
        root_cell_scale=0.65,
        tip_cell_scale=1.35,
    )
    print("two-skin graded hex-like wingbox lattice")
    print(f"nodes = {lat.nodes.shape[0]}")
    print(f"edges = {lat.edges.shape[0]}")
