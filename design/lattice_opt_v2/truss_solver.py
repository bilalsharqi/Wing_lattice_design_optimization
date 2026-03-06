
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrussResult:
    u: np.ndarray
    member_force: np.ndarray
    member_stress: np.ndarray
    tip_disp: float
    K: sp.csr_matrix
    lengths: np.ndarray
    dirs: np.ndarray


def assemble_truss_stiffness(nodes, edges, areas, E_modulus) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    N = nodes.shape[0]
    Ecount = edges.shape[0]
    if areas.shape[0] != Ecount:
        raise ValueError("areas must match number of edges")

    rows = []
    cols = []
    data = []

    lengths = np.zeros(Ecount)
    dirs = np.zeros((Ecount, 3))

    for e, (i, j) in enumerate(edges):
        xi = nodes[i]
        xj = nodes[j]
        d = xj - xi
        L = np.linalg.norm(d)
        if L < 1e-12:
            raise ValueError(f"Zero-length edge detected at e={e}, nodes=({i},{j})")
        n = d / L

        lengths[e] = L
        dirs[e] = n

        k = (E_modulus * areas[e] / L)
        nnT = np.outer(n, n)

        k11 = +k * nnT
        k12 = -k * nnT
        k21 = -k * nnT
        k22 = +k * nnT

        dofi = [3 * i, 3 * i + 1, 3 * i + 2]
        dofj = [3 * j, 3 * j + 1, 3 * j + 2]

        for a in range(3):
            for b in range(3):
                rows.append(dofi[a]); cols.append(dofi[b]); data.append(k11[a, b])
                rows.append(dofi[a]); cols.append(dofj[b]); data.append(k12[a, b])
                rows.append(dofj[a]); cols.append(dofi[b]); data.append(k21[a, b])
                rows.append(dofj[a]); cols.append(dofj[b]); data.append(k22[a, b])

    K = sp.coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()
    return K, lengths, dirs


def apply_dirichlet_bc(K: sp.csr_matrix, f: np.ndarray, fixed_dofs: np.ndarray):
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, np.unique(fixed_dofs))
    Kff = K[free][:, free].tocsr()
    ff = f[free]
    return Kff, ff, free


def solve_truss(nodes, edges, areas, E_modulus, f, fixed_dofs, tip_node_selector="max_y", regularization=1e-9):
    K, lengths, dirs = assemble_truss_stiffness(nodes, edges, areas, E_modulus)

    Kff, ff, free = apply_dirichlet_bc(K, f, fixed_dofs)

    # tiny regularization to help near-singular but still connected systems
    if regularization > 0.0 and Kff.shape[0] > 0:
        diag_scale = max(float(np.mean(Kff.diagonal())), 1.0)
        Kff = Kff + sp.eye(Kff.shape[0], format="csr") * (regularization * diag_scale)

    uf = spla.spsolve(Kff, ff)

    if np.any(~np.isfinite(uf)):
        raise np.linalg.LinAlgError("Non-finite solution detected in truss solve.")

    u = np.zeros(K.shape[0])
    u[free] = uf

    Ecount = edges.shape[0]
    member_force = np.zeros(Ecount)
    member_stress = np.zeros(Ecount)

    for e, (i, j) in enumerate(edges):
        ui = u[3 * i:3 * i + 3]
        uj = u[3 * j:3 * j + 3]
        axial = np.dot((uj - ui), dirs[e])
        N_e = (E_modulus * areas[e] / lengths[e]) * axial
        member_force[e] = N_e
        member_stress[e] = N_e / max(areas[e], 1e-16)

    if tip_node_selector == "max_y":
        tip_idx = int(np.argmax(nodes[:, 1]))
    else:
        raise ValueError("Only tip_node_selector='max_y' is supported.")

    tip_u = u[3 * tip_idx:3 * tip_idx + 3]
    tip_disp = float(np.linalg.norm(tip_u))

    return TrussResult(
        u=u,
        member_force=member_force,
        member_stress=member_stress,
        tip_disp=tip_disp,
        K=K,
        lengths=lengths,
        dirs=dirs,
    )
