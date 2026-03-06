
import numpy as np
import networkx as nx
from typing import Tuple


def build_graph(n_nodes: int, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(i), int(j)) for i, j in edges)
    return G


def _root_connected_subgraph(nodes_d, edges_d, wing):
    """
    Keep only the connected component that contains the root face nodes.
    If no root node remains, return an empty graph.
    """
    if len(nodes_d) == 0:
        return nodes_d, np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    root_nodes_local = np.where(wing.root_mask(nodes_d))[0]
    if len(root_nodes_local) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.array([], dtype=int)

    G = build_graph(len(nodes_d), edges_d)
    reachable = set()
    for r in root_nodes_local:
        reachable.add(int(r))
        reachable |= nx.node_connected_component(G, int(r))

    keep_local = np.array(sorted(list(reachable)), dtype=int)
    old2new = -np.ones(len(nodes_d), dtype=int)
    old2new[keep_local] = np.arange(len(keep_local))

    nodes_rc = nodes_d[keep_local]

    new_edges = []
    for i, j in edges_d:
        ni = old2new[i]
        nj = old2new[j]
        if ni >= 0 and nj >= 0:
            new_edges.append((ni, nj))

    edges_rc = np.array(new_edges, dtype=int) if len(new_edges) else np.zeros((0, 2), dtype=int)
    return nodes_rc, edges_rc, keep_local


def damage_by_khop(nodes, edges, seed, k=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove nodes within k hops of a seed node.
    Returns:
      nodes_d, edges_d, keep_old_indices, edge_keep_mask
    """
    G = build_graph(len(nodes), edges)

    remove = {int(seed)}
    frontier = {int(seed)}

    for _ in range(k):
        nxt = set()
        for v in frontier:
            nxt |= set(G.neighbors(v))
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
    return nodes_d, edges_d, keep, edge_keep_mask


def damage_by_nearest_n(nodes, edges, center_xyz, n_remove=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_xyz = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    d = np.linalg.norm(nodes - center_xyz, axis=1)
    remove = set(np.argsort(d)[:n_remove].tolist())

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
    return nodes_d, edges_d, keep, edge_keep_mask


def damage_and_keep_root_component(wing, nodes, edges, mode="khop", seed=None, k=2, center_xyz=None, n_remove=20):
    """
    Unified helper:
      1) apply damage
      2) keep only root-connected component
    Returns:
      nodes_rc, edges_rc, keep_old_indices_final, edge_keep_mask_final
    """
    if mode == "khop":
        nodes_d, edges_d, keep_old, edge_keep_mask = damage_by_khop(nodes, edges, seed=seed, k=k)
    elif mode == "nearest_n":
        nodes_d, edges_d, keep_old, edge_keep_mask = damage_by_nearest_n(nodes, edges, center_xyz=center_xyz, n_remove=n_remove)
    else:
        raise ValueError(f"Unknown damage mode '{mode}'")

    nodes_rc, edges_rc, keep_local = _root_connected_subgraph(nodes_d, edges_d, wing)

    if len(keep_local) == 0:
        return nodes_rc, edges_rc, np.array([], dtype=int), np.zeros(len(edges), dtype=bool)

    # map back to original kept nodes
    keep_old_final = keep_old[keep_local]

    # rebuild final edge mask relative to ORIGINAL edges
    keep_old_set = set(keep_old_final.tolist())
    edge_keep_mask_final = np.zeros(len(edges), dtype=bool)
    for e, (i, j) in enumerate(edges):
        if i in keep_old_set and j in keep_old_set:
            edge_keep_mask_final[e] = True

    return nodes_rc, edges_rc, keep_old_final, edge_keep_mask_final
