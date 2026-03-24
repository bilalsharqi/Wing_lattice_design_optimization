
import numpy as np
from beam_solver import solve_beam


def _expand_force_vector_to_beam(force_vector, n_nodes):
    f = np.asarray(force_vector, dtype=float).reshape(-1)
    if f.size == 6 * n_nodes:
        return f.copy()
    if f.size != 3 * n_nodes:
        raise ValueError(f"Expected 3N or 6N force vector for beam adapter, got {f.size}")
    fb = np.zeros(6 * n_nodes, dtype=float)
    fb.reshape(n_nodes, 6)[:, :3] = f.reshape(n_nodes, 3)
    return fb


def _expand_fixed_dofs_to_beam(fixed_dofs, n_nodes):
    fd = np.unique(np.asarray(fixed_dofs, dtype=int).reshape(-1))
    if fd.size == 0:
        return fd
    if np.max(fd) < 3 * n_nodes:
        nodes = np.unique(fd // 3)
        return (6 * nodes[:, None] + np.arange(6, dtype=int)[None, :]).ravel()
    return fd


def solve_truss(
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
    n_nodes = np.asarray(nodes).shape[0]
    fb = _expand_force_vector_to_beam(force_vector, n_nodes)
    fdb = _expand_fixed_dofs_to_beam(fixed_dofs, n_nodes)
    return solve_beam(
        nodes=nodes,
        edges=edges,
        areas=areas,
        e_modulus=E_modulus,
        force_vector=fb,
        fixed_dofs=fdb,
        wing=wing,
        regularization=regularization,
        tip_node_idx=tip_node_idx,
        nu=0.33,
    )
