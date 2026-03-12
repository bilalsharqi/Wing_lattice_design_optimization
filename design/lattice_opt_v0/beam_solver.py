
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


@dataclass
class BeamResult:
    u: np.ndarray
    member_force: np.ndarray
    member_stress: np.ndarray
    tip_disp: float
    tip_node_idx: int
    cond_est: float
    lengths: np.ndarray
    dirs: np.ndarray
    element_end_forces_local: np.ndarray


def _orthonormal_local_axes(ex: np.ndarray) -> np.ndarray:
    ex = np.asarray(ex, dtype=float)
    ex = ex / np.linalg.norm(ex)

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(ex, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    ey = np.cross(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    # rows = local basis expressed in global coordinates
    return np.vstack((ex, ey, ez))


def _frame_local_stiffness(E, G, A, Iy, Iz, J, L):
    k = np.zeros((12, 12), dtype=float)

    EA_L = E * A / L
    GJ_L = G * J / L

    EIy = E * Iy
    EIz = E * Iz

    # axial
    k[0, 0] = EA_L
    k[0, 6] = -EA_L
    k[6, 0] = -EA_L
    k[6, 6] = EA_L

    # torsion about local x
    k[3, 3] = GJ_L
    k[3, 9] = -GJ_L
    k[9, 3] = -GJ_L
    k[9, 9] = GJ_L

    # bending in local x-y plane (v, rz) -> uses EIz
    c1 = 12.0 * EIz / L**3
    c2 = 6.0 * EIz / L**2
    c3 = 4.0 * EIz / L
    c4 = 2.0 * EIz / L
    idx = [1, 5, 7, 11]
    sub = np.array([
        [ c1,  c2, -c1,  c2],
        [ c2,  c3, -c2,  c4],
        [-c1, -c2,  c1, -c2],
        [ c2,  c4, -c2,  c3],
    ])
    for a in range(4):
        for b in range(4):
            k[idx[a], idx[b]] += sub[a, b]

    # bending in local x-z plane (w, ry) -> uses EIy
    c1 = 12.0 * EIy / L**3
    c2 = 6.0 * EIy / L**2
    c3 = 4.0 * EIy / L
    c4 = 2.0 * EIy / L
    idx = [2, 4, 8, 10]
    sub = np.array([
        [ c1, -c2, -c1, -c2],
        [-c2,  c3,  c2,  c4],
        [-c1,  c2,  c1,  c2],
        [-c2,  c4,  c2,  c3],
    ])
    for a in range(4):
        for b in range(4):
            k[idx[a], idx[b]] += sub[a, b]

    return k


def _transformation_matrix(R):
    T = np.zeros((12, 12), dtype=float)
    for b in range(4):
        T[3*b:3*b+3, 3*b:3*b+3] = R
    return T


def _estimate_condition_symmetric(Kff):
    n = Kff.shape[0]
    if n == 0:
        return np.inf
    try:
        if n <= 350:
            vals = np.linalg.eigvalsh(Kff.toarray())
            lam_min = float(np.min(vals))
            lam_max = float(np.max(vals))
        else:
            lam_max = float(spla.eigsh(Kff, k=1, which="LA", return_eigenvectors=False)[0])
            lam_min = float(spla.eigsh(Kff, k=1, which="SA", return_eigenvectors=False)[0])
        if (not np.isfinite(lam_min)) or lam_min <= 0.0:
            return np.inf
        return lam_max / lam_min
    except Exception:
        try:
            return float(np.linalg.cond(Kff.toarray()))
        except Exception:
            return np.inf


def assemble_frame_stiffness(nodes, edges, areas, e_modulus, nu=0.33):
    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]
    ndof = 6 * n_nodes
    G = e_modulus / (2.0 * (1.0 + nu))

    rows, cols, data = [], [], []
    lengths = np.zeros(n_edges, dtype=float)
    dirs = np.zeros((n_edges, 3), dtype=float)

    k_local_all = []
    T_all = []

    for e, (i, j) in enumerate(edges):
        p = nodes[int(i)]
        q = nodes[int(j)]
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1e-12:
            raise ValueError(f"Zero-length beam element at edge {e}")

        ex = d / L
        R = _orthonormal_local_axes(ex)
        T = _transformation_matrix(R)

        A = float(areas[e])
        if A <= 0.0 or not np.isfinite(A):
            raise ValueError(f"Non-positive/invalid area at edge {e}: {A}")

        r = math.sqrt(A / math.pi)
        Iy = math.pi * r**4 / 4.0
        Iz = Iy
        J = math.pi * r**4 / 2.0

        k_local = _frame_local_stiffness(e_modulus, G, A, Iy, Iz, J, L)
        k_global = T.T @ k_local @ T

        dofs = [
            6 * int(i) + 0, 6 * int(i) + 1, 6 * int(i) + 2,
            6 * int(i) + 3, 6 * int(i) + 4, 6 * int(i) + 5,
            6 * int(j) + 0, 6 * int(j) + 1, 6 * int(j) + 2,
            6 * int(j) + 3, 6 * int(j) + 4, 6 * int(j) + 5,
        ]

        for a in range(12):
            for b in range(12):
                val = k_global[a, b]
                if val != 0.0:
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    data.append(val)

        lengths[e] = L
        dirs[e] = ex
        k_local_all.append(k_local)
        T_all.append(T)

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, lengths, dirs, k_local_all, T_all


def solve_beam(
    nodes,
    edges,
    areas,
    e_modulus,
    force_vector,
    fixed_dofs,
    wing=None,
    regularization=0.0,
    tip_node_idx=None,
    nu=0.33,
):
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    areas = np.asarray(areas, dtype=float)
    f = np.asarray(force_vector, dtype=float).reshape(-1)

    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes
    if f.size != ndof:
        raise ValueError(f"Beam solver expected force vector of size {ndof}, got {f.size}")

    K, lengths, dirs, k_local_all, T_all = assemble_frame_stiffness(nodes, edges, areas, e_modulus, nu=nu)

    if regularization > 0.0:
        K = K + regularization * sp.eye(ndof, format="csr")

    all_dofs = np.arange(ndof, dtype=int)
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    free = np.setdiff1d(all_dofs, fixed)

    Kff = K[free][:, free].tocsr()
    ff = f[free]

    cond_est = _estimate_condition_symmetric(Kff)

    uf = spla.spsolve(Kff, ff)
    u = np.zeros(ndof, dtype=float)
    u[free] = uf

    n_edges = edges.shape[0]
    element_end_forces_local = np.zeros((n_edges, 12), dtype=float)
    member_force = np.zeros((n_edges,), dtype=float)
    member_stress = np.zeros((n_edges,), dtype=float)

    for e, (i, j) in enumerate(edges):
        dofs = np.array([
            6 * int(i) + 0, 6 * int(i) + 1, 6 * int(i) + 2,
            6 * int(i) + 3, 6 * int(i) + 4, 6 * int(i) + 5,
            6 * int(j) + 0, 6 * int(j) + 1, 6 * int(j) + 2,
            6 * int(j) + 3, 6 * int(j) + 4, 6 * int(j) + 5,
        ], dtype=int)

        ue_global = u[dofs]
        T = T_all[e]
        k_local = k_local_all[e]
        ue_local = T @ ue_global
        fe_local = k_local @ ue_local
        element_end_forces_local[e, :] = fe_local

        A = float(areas[e])
        r = math.sqrt(A / math.pi)
        Iy = math.pi * r**4 / 4.0
        Iz = Iy
        J = math.pi * r**4 / 2.0
        c = r

        N = max(abs(fe_local[0]), abs(fe_local[6]))
        Tq = max(abs(fe_local[3]), abs(fe_local[9]))
        My = max(abs(fe_local[4]), abs(fe_local[10]))
        Mz = max(abs(fe_local[5]), abs(fe_local[11]))

        sigma_axial = N / A
        sigma_bend = c * math.sqrt((My / Iy) ** 2 + (Mz / Iz) ** 2)
        tau_torsion = (Tq * c / J) if J > 0.0 else 0.0
        sigma_vm = math.sqrt((sigma_axial + sigma_bend) ** 2 + 3.0 * tau_torsion ** 2)

        member_force[e] = N
        member_stress[e] = sigma_vm

    if tip_node_idx is None:
        tip_disp = float(np.linalg.norm(u[6*(n_nodes-1):6*(n_nodes-1)+3]))
        tip_node_idx = n_nodes - 1
    else:
        tip_disp = float(np.linalg.norm(u[6*int(tip_node_idx):6*int(tip_node_idx)+3]))

    return BeamResult(
        u=u,
        member_force=member_force,
        member_stress=member_stress,
        tip_disp=tip_disp,
        tip_node_idx=int(tip_node_idx),
        cond_est=cond_est,
        lengths=lengths,
        dirs=dirs,
        element_end_forces_local=element_end_forces_local,
    )
