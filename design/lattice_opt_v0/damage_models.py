import numpy as np
import networkx as nx


def build_graph(n_nodes: int, edges: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((int(i), int(j)) for i, j in edges)
    return g


def damage_by_khop(nodes, edges, seed, k=2):
    """
    Remove nodes within k graph hops of the seed node.
    Returns the damaged graph in reindexed form, plus bookkeeping.
    """
    g = build_graph(len(nodes), edges)
    remove = {int(seed)}
    frontier = {int(seed)}

    for _ in range(k):
        nxt = set()
        for v in frontier:
            nxt |= set(g.neighbors(v))
        remove |= nxt
        frontier = nxt

    keep = np.array([i for i in range(len(nodes)) if i not in remove], dtype=int)
    old2new = -np.ones(len(nodes), dtype=int)
    old2new[keep] = np.arange(len(keep))

    nodes_d = nodes[keep]

    edge_keep_mask = np.zeros(len(edges), dtype=bool)
    new_edges = []
    for e, (i, j) in enumerate(edges):
        if old2new[i] >= 0 and old2new[j] >= 0:
            new_edges.append((old2new[i], old2new[j]))
            edge_keep_mask[e] = True

    edges_d = np.array(new_edges, dtype=int) if len(new_edges) else np.zeros((0, 2), dtype=int)

    return nodes_d, edges_d, keep, edge_keep_mask, np.array(sorted(list(remove)), dtype=int)


def root_connected_component(nodes, edges, wing):
    """
    Keep only the component connected to at least one root-face node.
    """
    if len(nodes) == 0 or len(edges) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    root_nodes = np.where(wing.root_mask(nodes))[0]
    if len(root_nodes) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    g = build_graph(len(nodes), edges)
    reachable = set()
    for r in root_nodes:
        reachable |= nx.node_connected_component(g, int(r))

    keep = np.array(sorted(list(reachable)), dtype=int)
    old2new = -np.ones(len(nodes), dtype=int)
    old2new[keep] = np.arange(len(keep))

    nodes_rc = nodes[keep]

    new_edges = []
    for i, j in edges:
        ni = old2new[i]
        nj = old2new[j]
        if ni >= 0 and nj >= 0:
            new_edges.append((ni, nj))

    edges_rc = np.array(new_edges, dtype=int) if len(new_edges) else np.zeros((0, 2), dtype=int)

    return nodes_rc, edges_rc, keep


def peel_low_degree_nodes(nodes, edges, wing, min_degree=3):
    """
    Iteratively remove non-root nodes with degree < min_degree.
    This is a graph-screening heuristic, not a full structural stability proof.
    """
    if len(nodes) == 0 or len(edges) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    keep_global = np.arange(len(nodes), dtype=int)

    while True:
        g = build_graph(len(nodes), edges)
        root_nodes = set(np.where(wing.root_mask(nodes))[0].tolist())

        to_remove = []
        for v in g.nodes():
            if v in root_nodes:
                continue
            if g.degree(v) < min_degree:
                to_remove.append(v)

        if len(to_remove) == 0:
            break

        keep_local = np.array([i for i in range(len(nodes)) if i not in set(to_remove)], dtype=int)
        old2new = -np.ones(len(nodes), dtype=int)
        old2new[keep_local] = np.arange(len(keep_local))

        nodes = nodes[keep_local]

        new_edges = []
        for i, j in edges:
            ni = old2new[i]
            nj = old2new[j]
            if ni >= 0 and nj >= 0:
                new_edges.append((ni, nj))

        edges = np.array(new_edges, dtype=int) if len(new_edges) else np.zeros((0, 2), dtype=int)
        keep_global = keep_global[keep_local]

        if len(nodes) == 0 or len(edges) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    return nodes, edges, keep_global


def filter_to_root_connected_intact(nodes, edges, areas, wing):
    """
    For the intact graph after pruning:
    keep only the root-connected component.
    """
    nodes_rc, edges_rc, keep_local = root_connected_component(nodes, edges, wing)

    if len(keep_local) == 0:
        return nodes_rc, edges_rc, np.zeros((0,), dtype=float), keep_local, np.zeros(len(edges), dtype=bool)

    keep_set = set(keep_local.tolist())
    edge_keep_mask = np.zeros(len(edges), dtype=bool)
    for e, (i, j) in enumerate(edges):
        if i in keep_set and j in keep_set:
            edge_keep_mask[e] = True

    areas_rc = areas[edge_keep_mask]

    return nodes_rc, edges_rc, areas_rc, keep_local, edge_keep_mask


def damage_and_check_full_connectivity(wing, nodes, edges, areas, seed=None, k=2):
    """
    Hard rule:
      - apply damage
      - keep only root-connected component
      - peel low-degree non-root nodes as a graph screening heuristic
      - require the remaining damaged graph to be connected

    Important:
      graph_screen_passed != guaranteed mechanical stability.
      The solver must still verify that separately.
    """
    nodes_d, edges_d, keep_old, edge_keep_mask, removed_old = damage_by_khop(
        nodes, edges, seed=seed, k=k
    )

    if len(nodes_d) < 2 or len(edges_d) == 0:
        return {
            "connected": False,
            "stable_graph": False,
            "graph_screen_passed": False,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
            "areas_d": np.zeros((0,), dtype=float),
            "removed_nodes_original_idx": removed_old,
        }

    # keep only root-connected component
    nodes_d, edges_d, keep_local_rc = root_connected_component(nodes_d, edges_d, wing)
    if len(nodes_d) < 2 or len(edges_d) == 0:
        return {
            "connected": False,
            "stable_graph": False,
            "graph_screen_passed": False,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
            "areas_d": np.zeros((0,), dtype=float),
            "removed_nodes_original_idx": removed_old,
        }

    # peel low-degree nodes
    nodes_d, edges_d, keep_local_peel = peel_low_degree_nodes(
        nodes_d, edges_d, wing, min_degree=3
    )
    if len(nodes_d) < 2 or len(edges_d) == 0:
        return {
            "connected": False,
            "stable_graph": False,
            "graph_screen_passed": False,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
            "areas_d": np.zeros((0,), dtype=float),
            "removed_nodes_original_idx": removed_old,
        }

    g = build_graph(len(nodes_d), edges_d)
    connected = nx.is_connected(g)

    if not connected:
        return {
            "connected": False,
            "stable_graph": False,
            "graph_screen_passed": False,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
            "areas_d": np.zeros((0,), dtype=float),
            "removed_nodes_original_idx": removed_old,
        }

    # Rebuild surviving edge mask against the current intact graph using coordinates
    surviving_points = {tuple(np.round(p, 12)) for p in nodes_d}
    edge_keep_mask_final = np.zeros(len(edges), dtype=bool)
    for e, (i, j) in enumerate(edges):
        pi = tuple(np.round(nodes[i], 12))
        pj = tuple(np.round(nodes[j], 12))
        if pi in surviving_points and pj in surviving_points:
            edge_keep_mask_final[e] = True

    areas_d = areas[edge_keep_mask_final]

    return {
        "connected": True,
        "stable_graph": False,   # provisional only; real mechanical stability must be confirmed by solver
        "graph_screen_passed": True,
        "nodes_d": nodes_d,
        "edges_d": edges_d,
        "areas_d": areas_d,
        "removed_nodes_original_idx": removed_old,
    }