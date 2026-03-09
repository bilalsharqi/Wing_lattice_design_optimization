import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


@dataclass
class TrussResult:
    u: np.ndarray
    member_force: np.ndarray
    member_stress: np.ndarray
    tip_disp: float
    tip_node_idx: int
    cond_est: float
    lengths: np.ndarray
    dirs: np.ndarray


def assemble_truss_stiffness(nodes, edges, areas, e_modulus):
    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]

    rows, cols, data = [], [], []
    lengths = np.zeros(n_edges)
    dirs = np.zeros((n_edges, 3))

    for e, (i, j) in enumerate(edges):
        d = nodes[j] - nodes[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            raise ValueError(f"Zero-length edge at {e}")
        n = d / L
        lengths[e] = L
        dirs[e] = n

        k = (e_modulus * areas[e] / L)
        nnT = np.outer(n, n)
        k11 = +k * nnT
        k12 = -k * nnT

        dofi = [3 * i, 3 * i + 1, 3 * i + 2]
        dofj = [3 * j, 3 * j + 1, 3 * j + 2]

        for a in range(3):
            for b in range(3):
                rows.append(dofi[a]); cols.append(dofi[b]); data.append(k11[a, b])
                rows.append(dofi[a]); cols.append(dofj[b]); data.append(k12[a, b])
                rows.append(dofj[a]); cols.append(dofi[b]); data.append(k12[a, b])
                rows.append(dofj[a]); cols.append(dofj[b]); data.append(k11[a, b])

    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
    return K, lengths, dirs


def apply_dirichlet_bc(K, f, fixed_dofs):
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, np.unique(fixed_dofs))
    return K[free][:, free].tocsr(), f[free], free


def choose_tip_node(nodes, wing, connected_mask=None, span_fraction=0.95):
    """
    Choose a physically meaningful tip node from the connected load-carrying graph.
    Prefer nodes near the outboard end; among them choose the one with largest |z|.
    """
    if connected_mask is None:
        connected_mask = np.ones(len(nodes), dtype=bool)

    y = nodes[:, 1]
    y_max = np.max(y[connected_mask])
    candidates = np.where(connected_mask & (y >= span_fraction * y_max))[0]
    if len(candidates) == 0:
        candidates = np.where(connected_mask)[0]

    z_abs = np.abs(nodes[candidates, 2])
    return int(candidates[np.argmax(z_abs)])


def estimate_condition_number(Kff):
    """
    Cheap condition estimate from extreme eigenvalues.
    If it fails, return inf.
    """
    try:
        if Kff.shape[0] < 2:
            return np.inf
        # largest
        lam_max = spla.eigsh(Kff, k=1, which="LM", return_eigenvectors=False)[0]
        # smallest
        lam_min = spla.eigsh(Kff, k=1, which="SM", return_eigenvectors=False)[0]
        if lam_min <= 0:
            return np.inf
        return float(abs(lam_max / lam_min))
    except Exception:
        return np.inf


def solve_truss(nodes, edges, areas, e_modulus, f, fixed_dofs, wing, regularization=0.0, tip_node_idx=None):
    K, lengths, dirs = assemble_truss_stiffness(nodes, edges, areas, e_modulus)
    Kff, ff, free = apply_dirichlet_bc(K, f, fixed_dofs)

    if regularization > 0.0 and Kff.shape[0] > 0:
        diag_scale = max(float(np.mean(np.abs(Kff.diagonal()))), 1.0)
        Kff = Kff + sp.eye(Kff.shape[0], format="csr") * (regularization * diag_scale)

    cond_est = estimate_condition_number(Kff)

    uf = spla.spsolve(Kff, ff)
    if np.any(~np.isfinite(uf)):
        raise np.linalg.LinAlgError("Non-finite displacement vector")

    u = np.zeros(K.shape[0])
    u[free] = uf

    n_edges = edges.shape[0]
    member_force = np.zeros(n_edges)
    member_stress = np.zeros(n_edges)

    for e, (i, j) in enumerate(edges):
        ui = u[3 * i:3 * i + 3]
        uj = u[3 * j:3 * j + 3]
        axial = np.dot((uj - ui), dirs[e])
        N_e = (e_modulus * areas[e] / lengths[e]) * axial
        member_force[e] = N_e
        member_stress[e] = N_e / max(areas[e], 1e-16)

    if tip_node_idx is None:
        tip_node_idx = choose_tip_node(nodes, wing)

    tip_disp = float(np.linalg.norm(u[3 * tip_node_idx:3 * tip_node_idx + 3]))

    return TrussResult(
        u=u,
        member_force=member_force,
        member_stress=member_stress,
        tip_disp=tip_disp,
        tip_node_idx=tip_node_idx,
        cond_est=cond_est,
        lengths=lengths,
        dirs=dirs,
    )