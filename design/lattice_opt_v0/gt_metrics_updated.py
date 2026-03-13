import numpy as np
import networkx as nx


def build_graph(n_nodes: int, edges: np.ndarray, edge_weights: np.ndarray | None = None) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(n_nodes)))
    if edge_weights is None:
        g.add_edges_from((int(i), int(j)) for i, j in np.asarray(edges, dtype=int))
    else:
        for (i, j), w in zip(np.asarray(edges, dtype=int), np.asarray(edge_weights, dtype=float)):
            g.add_edge(int(i), int(j), weight=float(w))
    return g


def is_connected_safe(n_nodes: int, edges: np.ndarray) -> bool:
    if int(n_nodes) < 2:
        return False
    g = build_graph(n_nodes, edges)
    if g.number_of_edges() == 0:
        return False
    return nx.is_connected(g)


def algebraic_connectivity_safe(n_nodes: int, edges: np.ndarray) -> float:
    if not is_connected_safe(n_nodes, edges):
        return 0.0
    g = build_graph(n_nodes, edges)
    try:
        return float(nx.algebraic_connectivity(g))
    except Exception:
        return 0.0


def edge_betweenness_stats(n_nodes: int, edges: np.ndarray):
    g = build_graph(n_nodes, edges)
    if g.number_of_edges() == 0:
        return {"max": 0.0, "mean": 0.0, "var": 0.0}
    bc = nx.edge_betweenness_centrality(g, normalized=True)
    vals = np.array(list(bc.values()), dtype=float)
    return {
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "var": float(np.var(vals)),
    }


def edge_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if len(edges) == 0:
        return np.zeros((0,), dtype=float)
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    return np.linalg.norm(q - p, axis=1)


def root_boundary_nodes(nodes: np.ndarray, axis: int = 1, tol: float | None = None) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    vals = nodes[:, int(axis)]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = max(vmax - vmin, 1e-12)
    if tol is None:
        tol = max(1e-9, 1e-6 * span)
    return np.where(np.abs(vals - vmin) <= float(tol))[0]


def loaded_source_nodes(force_vector, n_nodes=None, dof_per_node=None, component=2, atol=1e-12):
    """
    Returns:
        idx : node indices with nonzero load in the selected component
        comp[idx] : corresponding absolute component magnitudes
    """
    f = np.asarray(force_vector, dtype=float).reshape(-1)

    if dof_per_node is None:
        if n_nodes is None:
            raise ValueError("Provide n_nodes or dof_per_node")
        if f.size % int(n_nodes) != 0:
            raise ValueError("force_vector size is not divisible by n_nodes")
        dof_per_node = f.size // int(n_nodes)

    if f.size % int(dof_per_node) != 0:
        raise ValueError("force_vector size must be divisible by dof_per_node")

    fmat = f.reshape((-1, int(dof_per_node)))
    comp = np.abs(fmat[:, int(component)])
    idx = np.where(comp > float(atol))[0]
    return idx, comp[idx]


def compute_edge_weights(nodes, edges, areas=None, E=1.0, mode="length"):
    mode = str(mode).lower()
    L = edge_lengths(nodes, edges)

    if mode == "length":
        return L

    if areas is None:
        raise ValueError(f"areas are required for edge_weight_mode='{mode}'")

    areas = np.asarray(areas, dtype=float).reshape(-1)
    if len(areas) != len(edges):
        raise ValueError("areas length must match number of edges")
    areas = np.clip(areas, 1e-16, None)

    if mode in ("area_scaled", "l_over_a", "length_over_area"):
        return L / areas

    if mode in ("compliance", "l_over_ea", "length_over_ea"):
        return L / (float(E) * areas)

    raise ValueError("edge_weight_mode must be one of: length, area_scaled, compliance")


def edge_boundary_betweenness_load_weighted(
    nodes,
    edges,
    source_nodes=None,
    target_nodes=None,
    force_vector=None,
    source_weights=None,
    edge_weight_mode="length",
    areas=None,
    E=1.0,
    normalized=True,
) -> np.ndarray:
    """
    Load-boundary edge betweenness centrality (EBC_LB), returned per edge in
    the same order as the input edge array.

    Default use case:
      - source_nodes: loaded nodes
      - target_nodes: root-clamped boundary nodes
      - edge weights: member lengths (or L/A, L/EA)

    If source_weights is provided, each source's contribution is weighted
    proportionally to its weight (e.g. nodal applied |Fz|).
    """
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)

    if len(edges) == 0:
        return np.zeros((0,), dtype=float)

    weights = compute_edge_weights(nodes, edges, areas=areas, E=E, mode=edge_weight_mode)
    g = build_graph(nodes.shape[0], edges, edge_weights=weights)

    if source_nodes is None:
        if force_vector is None:
            raise ValueError("Provide either source_nodes or force_vector.")
        source_nodes, inferred_w = loaded_source_nodes(
            force_vector,
            n_nodes=nodes.shape[0],
            dof_per_node=None,
            component=2,
        )
        if source_weights is None:
            source_weights = inferred_w

    source_nodes = np.asarray(source_nodes, dtype=int).reshape(-1)
    if target_nodes is None:
        target_nodes = root_boundary_nodes(nodes, axis=1)
    target_nodes = np.asarray(target_nodes, dtype=int).reshape(-1)

    # remove overlap between source and target sets, if any
    tset = set(map(int, target_nodes.tolist()))
    keep = np.array([int(s) not in tset for s in source_nodes], dtype=bool)
    source_nodes = source_nodes[keep]
    if source_weights is not None:
        source_weights = np.asarray(source_weights, dtype=float).reshape(-1)[keep]

    if len(source_nodes) == 0 or len(target_nodes) == 0:
        return np.zeros((len(edges),), dtype=float)

    if source_weights is None:
        source_weights = np.ones((len(source_nodes),), dtype=float)
    else:
        source_weights = np.asarray(source_weights, dtype=float).reshape(-1)
        if len(source_weights) != len(source_nodes):
            raise ValueError("source_weights must have same length as source_nodes")

    if np.all(source_weights <= 0.0):
        source_weights = np.ones_like(source_weights)

    source_weights = np.clip(source_weights, 0.0, None)
    source_weights = source_weights / max(np.sum(source_weights), 1e-16)

    accum = {}
    tgt = [int(t) for t in target_nodes.tolist()]
    for s, ws in zip(source_nodes.tolist(), source_weights.tolist()):
        bc_s = nx.edge_betweenness_centrality_subset(
            g,
            sources=[int(s)],
            targets=tgt,
            normalized=False,
            weight="weight",
        )
        for e, val in bc_s.items():
            accum[e] = accum.get(e, 0.0) + float(ws) * float(val)

    vals = np.zeros((len(edges),), dtype=float)
    for k, (i, j) in enumerate(edges):
        e = (int(i), int(j))
        er = (int(j), int(i))
        vals[k] = accum.get(e, accum.get(er, 0.0))

    if normalized and np.max(vals) > 0.0:
        vals = vals / np.max(vals)

    return vals


def edge_boundary_betweenness_stats(
    nodes,
    edges,
    source_nodes=None,
    target_nodes=None,
    force_vector=None,
    source_weights=None,
    edge_weight_mode="length",
    areas=None,
    E=1.0,
    normalized=True,
):
    vals = edge_boundary_betweenness_load_weighted(
        nodes=nodes,
        edges=edges,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        force_vector=force_vector,
        source_weights=source_weights,
        edge_weight_mode=edge_weight_mode,
        areas=areas,
        E=E,
        normalized=normalized,
    )
    if vals.size == 0:
        return {"max": 0.0, "mean": 0.0, "var": 0.0}
    return {
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "var": float(np.var(vals)),
    }