
import numpy as np
from typing import Dict


def distributed_vertical_load_to_nodes(
    nodes,
    span,
    total_lift_newtons,
    distribution="elliptic",
):
    """
    Map a spanwise distributed vertical load to nodal forces.
    y is spanwise. Load is applied in +z.
    """
    N = nodes.shape[0]
    f = np.zeros(3 * N)

    y = nodes[:, 1]
    eta = y / max(span, 1e-16)

    if distribution == "uniform":
        w = np.ones_like(eta)
    elif distribution == "elliptic":
        xi = 2.0 * eta - 1.0
        w = np.sqrt(np.clip(1.0 - xi * xi, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown distribution='{distribution}'")

    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("Load weights sum to zero.")

    Fz = total_lift_newtons * (w / wsum)
    f[2::3] = Fz
    return f


def nodal_force_lookup_table(nodes, lookup: Dict[int, np.ndarray]):
    N = nodes.shape[0]
    f = np.zeros(3 * N)
    for i, vec in lookup.items():
        f[3 * i:3 * i + 3] = np.asarray(vec, dtype=float)
    return f
