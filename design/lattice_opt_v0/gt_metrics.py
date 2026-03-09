import numpy as np
import networkx as nx


def build_graph(n_nodes: int, edges: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((int(i), int(j)) for i, j in edges)
    return g


def is_connected_safe(n_nodes: int, edges: np.ndarray) -> bool:
    if n_nodes < 2:
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