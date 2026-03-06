
import numpy as np
import networkx as nx


def build_graph(n_nodes: int, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(i), int(j)) for i, j in edges)
    return G


def algebraic_connectivity_safe(n_nodes: int, edges: np.ndarray) -> float:
    if n_nodes < 2:
        return 0.0
    G = build_graph(n_nodes, edges)
    if G.number_of_edges() == 0 or not nx.is_connected(G):
        return 0.0
    try:
        return float(nx.algebraic_connectivity(G))
    except Exception:
        return 0.0


def edge_betweenness_stats(n_nodes: int, edges: np.ndarray):
    G = build_graph(n_nodes, edges)
    if G.number_of_edges() == 0:
        return {"max": 0.0, "mean": 0.0, "var": 0.0}
    bc = nx.edge_betweenness_centrality(G, normalized=True)
    vals = np.array(list(bc.values()), dtype=float)
    return {
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "var": float(np.var(vals)),
    }
