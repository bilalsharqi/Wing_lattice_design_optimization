from dataclasses import dataclass
import numpy as np


@dataclass
class Lattice:
    nodes: np.ndarray
    edges: np.ndarray


def _grid_from_sizes(total_length: float, first_size: float, last_size: float, n_cells: int) -> np.ndarray:
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    sizes = np.linspace(first_size, last_size, n_cells)
    coords = np.concatenate(([0.0], np.cumsum(sizes)))
    coords *= total_length / coords[-1]
    return coords


def _xy_edges_from_grid(nx: int, ny: int, add_diagonals: bool):
    edges = []

    def nid(ix: int, iy: int) -> int:
        return iy * (nx + 1) + ix

    for iy in range(ny + 1):
        for ix in range(nx):
            edges.append((nid(ix, iy), nid(ix + 1, iy)))

    for iy in range(ny):
        for ix in range(nx + 1):
            edges.append((nid(ix, iy), nid(ix, iy + 1)))

    if add_diagonals:
        for iy in range(ny):
            for ix in range(nx):
                if (ix + iy) % 2 == 0:
                    edges.append((nid(ix, iy), nid(ix + 1, iy + 1)))
                else:
                    edges.append((nid(ix + 1, iy), nid(ix, iy + 1)))
    return edges


def generate_graded_hex_wingbox_lattice(
    span: float,
    chord: float,
    depth: float,
    ny: int = 14,
    nx: int = 6,
    root_cell_scale: float = 0.55,
    tip_cell_scale: float = 1.45,
    add_xy_diagonals: bool = True,
    add_verticals: bool = True,
    add_top_bottom_crosslinks: bool = True,
    add_xz_braces: bool = True,
    add_yz_braces: bool = True,
) -> Lattice:
    if span <= 0 or chord <= 0 or depth <= 0:
        raise ValueError("span, chord, and depth must be positive")
    if ny < 1 or nx < 1:
        raise ValueError("ny and nx must be >= 1")

    y_coords = _grid_from_sizes(span, root_cell_scale, tip_cell_scale, ny)
    x_coords = np.linspace(0.0, chord, nx + 1)

    nodes_bottom = []
    nodes_top = []
    for iy, y in enumerate(y_coords):
        x_offset = 0.5 * chord / nx if (iy % 2 == 1) else 0.0
        row_x = np.clip(x_coords + x_offset, 0.0, chord)
        row_x[0] = 0.0
        row_x[-1] = chord
        for x in row_x:
            nodes_bottom.append([x, y, -0.5 * depth])
            nodes_top.append([x, y,  0.5 * depth])

    nodes_bottom = np.asarray(nodes_bottom, dtype=float)
    nodes_top = np.asarray(nodes_top, dtype=float)
    nodes = np.vstack([nodes_bottom, nodes_top])
    n_layer = nodes_bottom.shape[0]

    def b(i: int) -> int:
        return i

    def t(i: int) -> int:
        return n_layer + i

    def nid(ix: int, iy: int) -> int:
        return iy * (nx + 1) + ix

    edges = set()
    xy_edges = _xy_edges_from_grid(nx, ny, add_diagonals=add_xy_diagonals)
    for i, j in xy_edges:
        edges.add(tuple(sorted((b(i), b(j)))))
        edges.add(tuple(sorted((t(i), t(j)))))

    if add_verticals:
        for i in range(n_layer):
            edges.add((b(i), t(i)))

    for iy in range(ny):
        for ix in range(nx):
            n00 = nid(ix, iy)
            n10 = nid(ix + 1, iy)
            n01 = nid(ix, iy + 1)
            n11 = nid(ix + 1, iy + 1)

            if add_top_bottom_crosslinks:
                edges.add(tuple(sorted((b(n00), t(n11)))))
                edges.add(tuple(sorted((b(n10), t(n01)))))

            if add_xz_braces:
                edges.add(tuple(sorted((b(n00), t(n10)))))
                edges.add(tuple(sorted((b(n01), t(n11)))))

            if add_yz_braces:
                edges.add(tuple(sorted((b(n00), t(n01)))))
                edges.add(tuple(sorted((b(n10), t(n11)))))

    edge_list = []
    for i, j in sorted(edges):
        if i != j and np.linalg.norm(nodes[j] - nodes[i]) > 1e-12:
            edge_list.append((i, j))

    return Lattice(nodes=nodes, edges=np.asarray(edge_list, dtype=int))


def generate_hex_wingbox_lattice(span: float, chord: float, depth: float, ny: int = 14, nx: int = 6, **kwargs) -> Lattice:
    return generate_graded_hex_wingbox_lattice(
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
    lat = generate_graded_hex_wingbox_lattice(span=4.0, chord=1.0, depth=0.16)
    print(f"nodes = {lat.nodes.shape[0]}")
    print(f"edges = {lat.edges.shape[0]}")
