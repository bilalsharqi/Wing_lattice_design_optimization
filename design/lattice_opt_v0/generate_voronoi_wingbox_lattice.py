import numpy as np
from dataclasses import dataclass

try:
    from scipy.spatial import Delaunay
except Exception as err:
    raise ImportError(
        "generate_voronoi_wingbox_lattice.py requires scipy.spatial.Delaunay. "
        f"Import failed with: {err}"
    )


@dataclass
class Lattice:
    nodes: np.ndarray
    edges: np.ndarray


def _add_edge(edge_set, a, b):
    a = int(a)
    b = int(b)
    if a == b:
        return
    i, j = sorted((a, b))
    edge_set.add((i, j))


def _clip_and_unique_points(points, chord, span, tol=1e-10):
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(chord))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(span))
    pts_round = np.round(pts / tol) * tol
    _, idx = np.unique(pts_round, axis=0, return_index=True)
    idx = np.sort(idx)
    return pts[idx]


def _sample_y_graded(n_points, span, root_density_scale=1.3, tip_density_scale=0.7, rng=None, n_bins=3000):
    """
    Sample spanwise locations with controllable root-to-tip grading.
    Larger density scale => more seeds locally.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y = np.linspace(0.0, float(span), int(max(200, n_bins)))
    s = y / max(float(span), 1e-12)
    density = (1.0 - s) * float(root_density_scale) + s * float(tip_density_scale)
    density = np.clip(density, 1e-12, None)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    u = rng.random(int(n_points))
    return np.interp(u, cdf, y)


def _generate_planform_seeds(
    n_seeds,
    chord,
    span,
    root_density_scale=1.3,
    tip_density_scale=0.7,
    rng=None,
    add_perimeter_frame=True,
    perimeter_nx=8,
    perimeter_ny=16,
):
    if rng is None:
        rng = np.random.default_rng(0)

    x = rng.random(int(n_seeds)) * float(chord)
    y = _sample_y_graded(
        int(n_seeds),
        span=float(span),
        root_density_scale=root_density_scale,
        tip_density_scale=tip_density_scale,
        rng=rng,
    )
    pts = np.column_stack((x, y))

    if add_perimeter_frame:
        xs = np.linspace(0.0, float(chord), int(max(2, perimeter_nx)))
        ys = np.linspace(0.0, float(span), int(max(2, perimeter_ny)))
        perimeter = []

        for xx in xs:
            perimeter.append([xx, 0.0])
            perimeter.append([xx, float(span)])

        for yy in ys:
            perimeter.append([0.0, yy])
            perimeter.append([float(chord), yy])

        perimeter.extend([
            [0.0, 0.0],
            [float(chord), 0.0],
            [0.0, float(span)],
            [float(chord), float(span)],
        ])

        pts = np.vstack((pts, np.asarray(perimeter, dtype=float)))

    pts = _clip_and_unique_points(pts, chord, span)
    return pts


def _delaunay_edges_2d(points_xy):
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.shape[0] < 3:
        return np.zeros((0, 2), dtype=int)

    tri = Delaunay(points_xy)
    edge_set = set()
    for simplex in tri.simplices:
        simplex = list(map(int, simplex))
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                _add_edge(edge_set, simplex[i], simplex[j])

    return np.asarray(sorted(edge_set), dtype=int)


def _delaunay_edges_3d(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.shape[0] < 4:
        return np.zeros((0, 2), dtype=int)

    tri = Delaunay(points_xyz)
    edge_set = set()
    for simplex in tri.simplices:
        simplex = list(map(int, simplex))
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                _add_edge(edge_set, simplex[i], simplex[j])

    return np.asarray(sorted(edge_set), dtype=int)


def _filter_short_edges(nodes, edges, min_edge_length):
    if len(edges) == 0:
        return edges
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    L = np.linalg.norm(q - p, axis=1)
    keep = L >= float(min_edge_length)
    return edges[keep]


def _nearest_neighbors_xy(points_xy, k=3):
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return [[] for _ in range(0)]
    dist2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    nbrs = []
    for i in range(n):
        order = np.argsort(dist2[i])
        nbrs.append([int(j) for j in order[1:1 + int(max(0, k))]])
    return nbrs


def generate_voronoi_wingbox_lattice(
    span,
    chord,
    depth,
    density_scale=1.0,
    root_density_scale=1.3,
    tip_density_scale=0.7,
    use_random_seed=False,
    random_seed=7,
    min_edge_length=0.03,
    add_perimeter_frame=True,
    add_verticals=True,
    add_web_diagonals=True,
    allow_extra_random_connectors=False,
    extra_connector_k=2,
    mode="wingbox",              # "wingbox" or "3d_stochastic"
    graph_type="delaunay",       # reserved for future extension
    chaos_level=0.0,             # 0 = plausible; larger => more jitter / connectors
    baseline_seed_count=90,
):
    """
    Stochastic lattice generator intended to plug into the optimizer like the
    square / graded-hex generators.

    Default mode ("wingbox"):
      - generates a graded stochastic XY graph
      - duplicates it at z=+depth/2 and z=-depth/2
      - adds optional vertical and diagonal web connectors

    Optional mode ("3d_stochastic"):
      - generates points throughout the wing volume
      - builds a 3D Delaunay graph directly

    Parameters
    ----------
    density_scale : float
        Rough complexity control. ~1.0 targets a graph complexity comparable to
        existing periodic cases; >1 creates denser graphs.
    root_density_scale, tip_density_scale : float
        Control seed density grading from root to tip.
    use_random_seed : bool
        If False, the generator is reproducible using random_seed.
        If True, a fresh RNG seed is used each run.
    min_edge_length : float
        Absolute minimum edge length [m] retained in the graph.
    """
    if graph_type.lower() != "delaunay":
        raise ValueError("First implementation supports graph_type='delaunay' only.")

    if use_random_seed:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(random_seed))

    mode = mode.lower().strip()
    chaos_level = float(max(0.0, chaos_level))
    n_base = int(max(12, round(float(baseline_seed_count) * float(density_scale))))

    if mode == "wingbox":
        pts_xy = _generate_planform_seeds(
            n_base,
            chord=float(chord),
            span=float(span),
            root_density_scale=float(root_density_scale),
            tip_density_scale=float(tip_density_scale),
            rng=rng,
            add_perimeter_frame=bool(add_perimeter_frame),
            perimeter_nx=max(6, int(round(8 * max(1.0, density_scale)))),
            perimeter_ny=max(10, int(round(16 * max(1.0, density_scale)))),
        )

        if chaos_level > 0.0:
            jitter_x = 0.03 * float(chord) * chaos_level
            jitter_y = 0.03 * float(span) * chaos_level
            jitter = np.column_stack((
                rng.normal(scale=jitter_x, size=pts_xy.shape[0]),
                rng.normal(scale=jitter_y, size=pts_xy.shape[0]),
            ))
            pts_xy = _clip_and_unique_points(pts_xy + jitter, chord, span)

        skin_edges = _delaunay_edges_2d(pts_xy)

        top_nodes = np.column_stack((pts_xy[:, 0], pts_xy[:, 1], +0.5 * float(depth) * np.ones(len(pts_xy))))
        bot_nodes = np.column_stack((pts_xy[:, 0], pts_xy[:, 1], -0.5 * float(depth) * np.ones(len(pts_xy))))
        nodes = np.vstack((top_nodes, bot_nodes))
        n_skin = pts_xy.shape[0]

        edge_set = set()

        # top and bottom skins
        for i, j in skin_edges:
            _add_edge(edge_set, i, j)
            _add_edge(edge_set, i + n_skin, j + n_skin)

        # direct vertical connectors
        if add_verticals:
            for i in range(n_skin):
                _add_edge(edge_set, i, i + n_skin)

        # web diagonals between top and bottom skins
        if add_web_diagonals:
            nbrs = _nearest_neighbors_xy(pts_xy, k=max(int(extra_connector_k), int(3 + round(2 * chaos_level))))
            for i in range(n_skin):
                for j in nbrs[i]:
                    _add_edge(edge_set, i, j + n_skin)
                    _add_edge(edge_set, i + n_skin, j)

        # optional extra random connectors to increase stochasticity
        if allow_extra_random_connectors:
            n_extra = int(round((1.0 + 2.0 * chaos_level) * 0.15 * n_skin))
            for _ in range(max(0, n_extra)):
                i = int(rng.integers(0, n_skin))
                j = int(rng.integers(0, n_skin))
                if i == j:
                    continue
                a = i if rng.random() < 0.5 else i + n_skin
                b = j if rng.random() < 0.5 else j + n_skin
                _add_edge(edge_set, a, b)

        edges = np.asarray(sorted(edge_set), dtype=int)
        edges = _filter_short_edges(nodes, edges, float(min_edge_length))

        # Re-enforce perimeter frame explicitly to guarantee clean boundaries
        if add_perimeter_frame:
            tol = 1e-9
            le = np.where(np.isclose(pts_xy[:, 0], 0.0, atol=tol))[0]
            te = np.where(np.isclose(pts_xy[:, 0], float(chord), atol=tol))[0]
            root = np.where(np.isclose(pts_xy[:, 1], 0.0, atol=tol))[0]
            tip = np.where(np.isclose(pts_xy[:, 1], float(span), atol=tol))[0]

            def chain_by(arr, axis):
                arr = list(map(int, arr))
                arr = sorted(arr, key=lambda ii: pts_xy[ii, axis])
                return arr

            edge_set = set(map(tuple, edges.tolist()))
            for chain in [chain_by(le, 1), chain_by(te, 1), chain_by(root, 0), chain_by(tip, 0)]:
                for a, b in zip(chain[:-1], chain[1:]):
                    _add_edge(edge_set, a, b)
                    _add_edge(edge_set, a + n_skin, b + n_skin)

            edges = np.asarray(sorted(edge_set), dtype=int)

        return Lattice(nodes=nodes, edges=edges)

    elif mode == "3d_stochastic":
        # stochastic points throughout the wing volume
        x = rng.random(n_base) * float(chord)
        y = _sample_y_graded(
            n_base,
            span=float(span),
            root_density_scale=float(root_density_scale),
            tip_density_scale=float(tip_density_scale),
            rng=rng,
        )
        z = (rng.random(n_base) - 0.5) * float(depth)

        pts = np.column_stack((x, y, z))

        if add_perimeter_frame:
            # add a sparse box-like boundary scaffold
            xs = np.linspace(0.0, float(chord), 5)
            ys = np.linspace(0.0, float(span), 9)
            zs = np.array([-0.5 * float(depth), +0.5 * float(depth)])
            frame = []
            for xx in xs:
                for zz in zs:
                    frame.append([xx, 0.0, zz])
                    frame.append([xx, float(span), zz])
            for yy in ys:
                for zz in zs:
                    frame.append([0.0, yy, zz])
                    frame.append([float(chord), yy, zz])
            pts = np.vstack((pts, np.asarray(frame, dtype=float)))

        if chaos_level > 0.0:
            pts[:, 0] += rng.normal(scale=0.03 * float(chord) * chaos_level, size=pts.shape[0])
            pts[:, 1] += rng.normal(scale=0.03 * float(span) * chaos_level, size=pts.shape[0])
            pts[:, 2] += rng.normal(scale=0.03 * float(depth) * chaos_level, size=pts.shape[0])

        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(chord))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(span))
        pts[:, 2] = np.clip(pts[:, 2], -0.5 * float(depth), +0.5 * float(depth))

        nodes = pts
        edges = _delaunay_edges_3d(nodes)
        edges = _filter_short_edges(nodes, edges, float(min_edge_length))
        return Lattice(nodes=nodes, edges=edges)

    else:
        raise ValueError("mode must be 'wingbox' or '3d_stochastic'")


if __name__ == "__main__":
    lat = generate_voronoi_wingbox_lattice(
        span=4.0,
        chord=1.0,
        depth=0.16,
        density_scale=1.0,
        root_density_scale=1.3,
        tip_density_scale=0.7,
        use_random_seed=False,
        random_seed=7,
        min_edge_length=0.03,
        add_perimeter_frame=True,
        add_verticals=True,
        add_web_diagonals=True,
        allow_extra_random_connectors=False,
        mode="wingbox",
        graph_type="delaunay",
        chaos_level=0.0,
    )
    print("nodes:", lat.nodes.shape)
    print("edges:", lat.edges.shape)
    print("x-range:", lat.nodes[:, 0].min(), lat.nodes[:, 0].max())
    print("y-range:", lat.nodes[:, 1].min(), lat.nodes[:, 1].max())
    print("z-range:", lat.nodes[:, 2].min(), lat.nodes[:, 2].max())