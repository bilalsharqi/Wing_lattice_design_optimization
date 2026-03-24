import importlib
import numpy as np

from load_mapping import distributed_vertical_load_to_nodes


def get_solver_function(backend_name):
    backend_name = backend_name.lower().strip()
    if backend_name == "truss":
        return importlib.import_module("truss_solver").solve_truss
    if backend_name == "responsegt":
        return importlib.import_module("ResGT_optimizer_adapter").solve_truss
    if backend_name == "beam":
        return importlib.import_module("beam_optimizer_adapter").solve_truss
    raise ValueError(f"Unknown solver backend '{backend_name}'")


def extract_translation_u(u: np.ndarray, n_nodes: int) -> np.ndarray:
    """Return (n_nodes,3) translations from either 3N or 6N DOF vectors."""
    u = np.asarray(u).reshape(-1)
    if u.size == 3 * n_nodes:
        return u.reshape(n_nodes, 3)
    if u.size == 6 * n_nodes:
        return u.reshape(n_nodes, 6)[:, :3]
    raise ValueError(
        f"Unexpected displacement size {u.size}; expected {3*n_nodes} (truss) or {6*n_nodes} (beam)"
    )


def build_force_vector(nodes, wing, total_lift, backend_name, physics_settings, distribution="elliptic"):
    """
    Build solver load vector.
    - truss / responsegt: 3N translational force vector
    - beam: 6N vector with same translational forces plus a spanwise torsional moment
      My = -(elastic_axis_x - x_cp) * Fz
    """
    f3 = distributed_vertical_load_to_nodes(nodes, wing.span, total_lift, distribution=distribution)
    if backend_name.lower().strip() != "beam":
        return f3

    n_nodes = nodes.shape[0]
    f6 = np.zeros(6 * n_nodes, dtype=float)
    f6.reshape(n_nodes, 6)[:, :3] = f3.reshape(n_nodes, 3)

    if physics_settings.get("excite_torsion", False):
        elastic_axis_x = float(physics_settings.get("elastic_axis_x_frac", 0.35)) * float(wing.chord)
        x_cp = float(physics_settings.get("x_cp_frac", 0.25)) * float(wing.chord)
        dx = elastic_axis_x - x_cp
        f6.reshape(n_nodes, 6)[:, 4] += -dx * f6.reshape(n_nodes, 6)[:, 2]
    return f6
