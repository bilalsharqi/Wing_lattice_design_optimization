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

    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    d = q - p
    lengths = np.linalg.norm(d, axis=1)
    if np.any(lengths < 1e-12):
        bad = int(np.where(lengths < 1e-12)[0][0])
        raise ValueError(f"Zero-length edge at {bad}")
    dirs = d / lengths[:, None]

    kfac = (e_modulus * areas / lengths)[:, None, None]
    nnT = dirs[:, :, None] * dirs[:, None, :]
    k11 = kfac * nnT
    k12 = -k11

    dofi = np.stack([3 * edges[:, 0], 3 * edges[:, 0] + 1, 3 * edges[:, 0] + 2], axis=1)
    dofj = np.stack([3 * edges[:, 1], 3 * edges[:, 1] + 1, 3 * edges[:, 1] + 2], axis=1)

    row_ii = np.repeat(dofi, 3, axis=1)
    col_ii = np.tile(dofi, (1, 3))
    row_ij = np.repeat(dofi, 3, axis=1)
    col_ij = np.tile(dofj, (1, 3))
    row_ji = np.repeat(dofj, 3, axis=1)
    col_ji = np.tile(dofi, (1, 3))
    row_jj = np.repeat(dofj, 3, axis=1)
    col_jj = np.tile(dofj, (1, 3))

    rows = np.concatenate([row_ii, row_ij, row_ji, row_jj], axis=1).ravel()
    cols = np.concatenate([col_ii, col_ij, col_ji, col_jj], axis=1).ravel()
    data = np.concatenate(
        [k11.reshape(n_edges, 9), k12.reshape(n_edges, 9), k12.reshape(n_edges, 9), k11.reshape(n_edges, 9)],
        axis=1,
    ).ravel()

    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
    return K, lengths, dirs


def apply_dirichlet_bc(K, f, fixed_dofs):
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    mask = np.ones(K.shape[0], dtype=bool)
    mask[fixed] = False
    free = np.where(mask)[0]
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

    ui = u.reshape(-1, 3)[edges[:, 0]]
    uj = u.reshape(-1, 3)[edges[:, 1]]
    axial = np.einsum("ij,ij->i", (uj - ui), dirs)
    member_force = (e_modulus * areas / lengths) * axial
    member_stress = member_force / np.maximum(areas, 1e-16)

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
