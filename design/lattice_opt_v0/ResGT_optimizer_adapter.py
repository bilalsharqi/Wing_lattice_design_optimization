
import numpy as np
from dataclasses import dataclass
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, eigsh


@dataclass
class ResponseGTResult:
    u: np.ndarray
    member_force: np.ndarray
    member_stress: np.ndarray
    cond_est: float
    tip_disp: float


def build_compatibility_matrix(nodes, edges):
    nodes = np.asarray(nodes)
    edges = np.asarray(edges)

    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]

    rows = []
    cols = []
    data = []

    lengths = np.zeros(n_edges)

    for e, (i, j) in enumerate(edges):
        xi = nodes[i]
        xj = nodes[j]

        d = xj - xi
        L = np.linalg.norm(d)
        if L == 0:
            raise ValueError("Zero-length edge detected")

        dir_vec = d / L
        lengths[e] = L

        for k in range(3):
            rows.append(e)
            cols.append(3 * i + k)
            data.append(-dir_vec[k])

            rows.append(e)
            cols.append(3 * j + k)
            data.append(dir_vec[k])

    B = coo_matrix((data, (rows, cols)), shape=(n_edges, 3 * n_nodes)).tocsr()

    return B, lengths


def solve_responsegt(
    nodes,
    edges,
    areas,
    E_modulus,
    force_vector,
    fixed_dofs,
    wing=None,
    regularization=0.0,
    tip_node_idx=None,
):

    nodes = np.asarray(nodes)
    edges = np.asarray(edges)
    areas = np.asarray(areas)
    f = np.asarray(force_vector).reshape(-1)

    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]
    ndof = 3 * n_nodes

    B, lengths = build_compatibility_matrix(nodes, edges)

    axial_stiffness = (E_modulus * areas) / lengths

    K_local = csr_matrix(
        (axial_stiffness, (np.arange(n_edges), np.arange(n_edges))),
        shape=(n_edges, n_edges),
    )

    K = B.T @ K_local @ B

    if regularization > 0:
        K = K + regularization * csr_matrix(np.eye(ndof))

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    Kff = K[free_dofs][:, free_dofs]
    ff = f[free_dofs]

    uf = spsolve(Kff, ff)

    u = np.zeros(ndof)
    u[free_dofs] = uf

    strain = B @ u
    member_force = axial_stiffness * strain
    member_stress = member_force / areas

    try:
        if Kff.shape[0] < 2:
            cond_est = np.inf
        else:
            lam_max = float(eigsh(Kff, k=1, which="LM", return_eigenvectors=False)[0])
            lam_min = float(eigsh(Kff, k=1, which="SM", return_eigenvectors=False)[0])
            cond_est = (lam_max / lam_min) if lam_min > 0.0 else np.inf
    except Exception:
        cond_est = np.inf

    tip_disp = 0.0
    if tip_node_idx is not None:
        tip_u = u[3 * tip_node_idx : 3 * tip_node_idx + 3]
        tip_disp = np.linalg.norm(tip_u)

    return ResponseGTResult(
        u=u,
        member_force=member_force,
        member_stress=member_stress,
        cond_est=cond_est,
        tip_disp=tip_disp,
    )


# alias so optimizer can import solve_truss
solve_truss = solve_responsegt
