# CONFIG-DRIVEN optimizer with settings header
# Generated from existing workflow-preserving solver-switch driver.

import os
import json
import shutil
import importlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import imageio.v2 as imageio

from wingbox_domain import WingBox
from load_mapping import distributed_vertical_load_to_nodes
from truss_solver import choose_tip_node
from damage_models import filter_to_root_connected_intact, damage_and_check_full_connectivity
from gt_metrics import is_connected_safe, algebraic_connectivity_safe, edge_betweenness_stats

from generate_octet_lattice import generate_octet_ground_structure
from generate_square_lattice import generate_square_wingbox_lattice
from grouped_pruning import build_spanwise_bay_groups, evaluate_group_scores, prune_one_group
from export_lattice_to_nastran_bdf import export_lattice_to_nastran


# ======================================================================
# USER SETTINGS
# ======================================================================
# Edit this section only. The rest of the script reads from `settings`.
#
# Supported lattice types
# -----------------------
# square:
#   nx, ny, nz
#   root_cell_scale, tip_cell_scale   # optional grading; set both to 1.0 for uniform square
#   add_xy_diagonals, add_yz_diagonals, add_xz_diagonals
#
# graded_hex
#   Uses lattice[nx], lattice[ny]
#   Uses lattice[root_cell_scale], lattice[tip_cell_scale]
#
# two_skin_graded_hex
#   Uses lattice[nx], lattice[ny]
#   Uses lattice[root_cell_scale], lattice[tip_cell_scale]
#   Uses lattice[add_skin_diagonals], lattice[add_verticals],
#        lattice[add_yz_diagonals], lattice[add_xz_diagonals],
#        lattice[add_perimeter_frame]
#
# octet
#   Uses lattice[nx], lattice[ny], lattice[nz]
#
# voronoi:
#   density_scale
#   root_density_scale, tip_density_scale
#   use_random_seed, random_seed
#   min_edge_length
#   add_perimeter_frame
#   add_verticals, add_web_diagonals
#   allow_extra_random_connectors, extra_connector_k
#   voronoi_mode = "wingbox" or "3d_stochastic"
#   graph_type = "delaunay"
#   chaos_level
#   baseline_seed_count
#
# Supported solver backends
# -------------------------
# truss
# responsegt
# beam
#
# Output behavior
# ---------------
# Each run is saved to:
#   results/<run_name>_<timestamp>/
# with backend-specific subfolders for data, plots, frames, and FEM exports.
# ======================================================================

settings = {
    "run": {
        "run_name": "lattice_opt",
        "lattice_type": "graded_hex",     # square, graded_hex, two_skin_graded_hex, octet
        "backends": ["truss", "responsegt", "beam"],
    },

    "solvers": {
        "E_modulus": 70e9,
        "density": 2700.0,
        "sigma_allow": 250e6,
        "u_tip_max": 0.25,
        "cond_max": 1e7,
    },

    "wing": {
        "span": 4.0,
        "chord": 1.0,
        "depth": 0.16,
        "root_tol": 1e-9,
        "m_vehicle": 50.0,
    },

    "physics": {
        "g": 9.80665,
        "load_factor": 5.0,
        "excite_torsion": True,
        "elastic_axis_x_frac": 0.25,
        "x_cp_frac": 0.35,
    },

    "optimization": {
        "enable_area_sizing": False,
        "a_min": 5e-7,
        "a_max": 5e-4,
        "a_init": 5e-5,
        "max_iterations": 25,
        "prune_after_iter": 1,
        "shrink_factor": 0.970,
        "grow_factor": 1.08,
    },

    "damage": {
        "damage_mode": "local_edge_ring",   # local_edge_ring or legacy_graph_hops
        "damage_k": 0,
        "seed_span_fraction": 0.60,
        "seed_x_fraction": 0.50,
    },

    "lattice": {
        "nx": 3,
        "ny": 14,
        "nz": 2,
        "root_cell_scale": 0.5,
        "tip_cell_scale": 1.8,
        "add_xy_diagonals": True,
        "add_yz_diagonals": True,
        "add_xz_diagonals": True,
        "add_skin_diagonals": True,
        "add_verticals": True,
        "add_perimeter_frame": True,
        "density_scale": 1.0, # here onwards lattice settings are for voronoi
        "root_density_scale": 1.3,
        "tip_density_scale": 0.7,
        "use_random_seed": False,
        "random_seed": 7,
        "min_edge_length": 0.05,
        "add_perimeter_frame": True,
        "add_verticals": True,
        "add_web_diagonals": True,
        "allow_extra_random_connectors": False,
        "extra_connector_k": 2,
        "voronoi_mode": "wingbox",     # "wingbox" or "3d_stochastic"
        "graph_type": "delaunay",
        "chaos_level": 0.0,
        "baseline_seed_count": 90,
    },

    "output": {
        "results_dir": "results",
        "save_frames": True,
        "fps": 2,
        "save_hdf5": True,
        "show_deformed_shape": True,
        "deformation_scale": 3.0,
        "plot_undeformed_background": True,
        "export_final_fem": True,
        "fem_element_length": 0.03,   # meters (3 cm default)
        "fem_include_loads": False,   # match the loads applied here
    },
}


def make_results_paths(base_dir, run_name="lattice_opt"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_dir, f"{run_name}_{ts}")
    paths = {
        "root": root,
        "data": os.path.join(root, "data"),
        "plots": os.path.join(root, "plots"),
        "frames": os.path.join(root, "frames"),
        "fem": os.path.join(root, "fem"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def make_fixed_dofs_for_root_clamp(nodes, root_mask):
    fixed_nodes = np.where(root_mask)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes + 0, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    return np.unique(fixed_dofs)


def estimate_mass(nodes, edges, areas, density):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    L = np.linalg.norm(q - p, axis=1)
    return float(np.sum(density * L * areas))


def simple_sizing_update(
    areas,
    stress,
    sigma_allow,
    a_min,
    a_max,
    shrink_factor=0.970,
    grow_factor=1.08,
):
    ratio = np.abs(stress) / max(sigma_allow, 1e-16)
    new_a = areas.copy()
    new_a[ratio > 1.0] *= grow_factor
    new_a[ratio < 0.20] *= shrink_factor
    return np.clip(new_a, a_min, a_max)


def build_lattice(wing, lattice_type, lattice_settings):
    if lattice_type == "square":
        lat = generate_square_wingbox_lattice(
            span=wing.span,
            chord=wing.chord,
            depth=wing.depth,
            ny=lattice_settings["ny"],
            nx=lattice_settings["nx"],
            nz=lattice_settings["nz"],
            root_cell_scale=lattice_settings["root_cell_scale"],
            tip_cell_scale=lattice_settings["tip_cell_scale"],
            add_xy_diagonals=lattice_settings["add_xy_diagonals"],
            add_yz_diagonals=lattice_settings["add_yz_diagonals"],
            add_xz_diagonals=lattice_settings["add_xz_diagonals"],
        )
    elif lattice_type == "octet":
        lat = generate_octet_ground_structure(
            wing.span,
            wing.chord,
            wing.depth,
            ny=lattice_settings["ny"],
            nx=lattice_settings["nx"],
            nz=lattice_settings["nz"],
        )
    elif lattice_type == "graded_hex":
        from generate_graded_hex_wingbox_lattice import generate_graded_hex_wingbox_lattice
        lat = generate_graded_hex_wingbox_lattice(
            span=wing.span,
            chord=wing.chord,
            depth=wing.depth,
            ny=lattice_settings["ny"],
            nx=lattice_settings["nx"],
            root_cell_scale=lattice_settings["root_cell_scale"],
            tip_cell_scale=lattice_settings["tip_cell_scale"],
        )
    elif lattice_type == "two_skin_graded_hex":
        from generate_two_skin_graded_hex_wingbox_lattice import (
            generate_two_skin_graded_hex_wingbox_lattice,
        )

        lat = generate_two_skin_graded_hex_wingbox_lattice(
            span=wing.span,
            chord=wing.chord,
            depth=wing.depth,
            ny=lattice_settings["ny"],
            nx=lattice_settings["nx"],
            root_cell_scale=lattice_settings["root_cell_scale"],
            tip_cell_scale=lattice_settings["tip_cell_scale"],
            add_skin_diagonals=lattice_settings["add_skin_diagonals"],
            add_verticals=lattice_settings["add_verticals"],
            add_yz_diagonals=lattice_settings["add_yz_diagonals"],
            add_xz_diagonals=lattice_settings["add_xz_diagonals"],
            add_perimeter_frame=lattice_settings["add_perimeter_frame"],
        )
    elif lattice_type == "voronoi":
        from generate_voronoi_wingbox_lattice import generate_voronoi_wingbox_lattice
    
        lat = generate_voronoi_wingbox_lattice(
            span=wing.span,
            chord=wing.chord,
            depth=wing.depth,
            density_scale=lattice_settings["density_scale"],
            root_density_scale=lattice_settings["root_density_scale"],
            tip_density_scale=lattice_settings["tip_density_scale"],
            use_random_seed=lattice_settings["use_random_seed"],
            random_seed=lattice_settings["random_seed"],
            min_edge_length=lattice_settings["min_edge_length"],
            add_perimeter_frame=lattice_settings["add_perimeter_frame"],
            add_verticals=lattice_settings["add_verticals"],
            add_web_diagonals=lattice_settings["add_web_diagonals"],
            allow_extra_random_connectors=lattice_settings["allow_extra_random_connectors"],
            extra_connector_k=lattice_settings["extra_connector_k"],
            mode=lattice_settings["voronoi_mode"],
            graph_type=lattice_settings["graph_type"],
            chaos_level=lattice_settings["chaos_level"],
            baseline_seed_count=lattice_settings["baseline_seed_count"],
        )
    else:
        raise ValueError(f"Unknown lattice_type='{lattice_type}'")
    return lat.nodes, lat.edges


def save_iteration_hdf5(h5, it, data_dict):
    grp = h5.create_group(f"iter_{it:04d}")
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            grp.create_dataset(key, data=val, compression="gzip")
        elif np.isscalar(val) or isinstance(val, (str, bytes, bool)):
            grp.attrs[key] = val
        elif isinstance(val, dict):
            sub = grp.create_group(key)
            for k2, v2 in val.items():
                if np.isscalar(v2) or isinstance(v2, (str, bytes, bool)):
                    sub.attrs[k2] = v2
                else:
                    sub.create_dataset(k2, data=np.asarray(v2), compression="gzip")
        elif val is None:
            grp.attrs[key] = "None"
        else:
            grp.attrs[key] = json.dumps(val)


def _write_h5_value(parent, key, val):
    """Recursively write nested settings/metadata into HDF5."""
    if isinstance(val, dict):
        sub = parent.create_group(str(key))
        for k2, v2 in val.items():
            _write_h5_value(sub, k2, v2)
        return

    if val is None:
        parent.attrs[str(key)] = "None"
        return

    if isinstance(val, (str, bytes, bool)) or np.isscalar(val):
        parent.attrs[str(key)] = val
        return

    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            parent.create_dataset(str(key), data=np.asarray([], dtype=float))
            return
        if all(isinstance(x, (str, bytes)) for x in val):
            dt = h5py.string_dtype(encoding="utf-8")
            parent.create_dataset(str(key), data=np.asarray(val, dtype=dt))
            return
        if all(isinstance(x, (bool, np.bool_)) for x in val):
            parent.create_dataset(str(key), data=np.asarray(val, dtype=bool))
            return
        if all(np.isscalar(x) for x in val):
            parent.create_dataset(str(key), data=np.asarray(val))
            return
        parent.attrs[str(key)] = json.dumps(val)
        return

    if isinstance(val, np.ndarray):
        if val.dtype.kind in {"U", "O"}:
            try:
                dt = h5py.string_dtype(encoding="utf-8")
                parent.create_dataset(str(key), data=val.astype(dt))
            except Exception:
                parent.attrs[str(key)] = json.dumps(val.tolist())
        else:
            parent.create_dataset(str(key), data=val, compression="gzip")
        return

    parent.attrs[str(key)] = json.dumps(val)


def write_settings_meta(meta_group, settings, results_paths=None, backend_name=None, lattice_type=None):
    """Write full run settings and context into /meta."""
    if backend_name is not None:
        meta_group.attrs["solver_backend"] = backend_name
    if lattice_type is not None:
        meta_group.attrs["lattice_type"] = lattice_type

    meta_group.attrs["settings_json"] = json.dumps(settings, indent=2)

    settings_group = meta_group.create_group("settings")
    for section_name, section_val in settings.items():
        _write_h5_value(settings_group, section_name, section_val)

    if results_paths is not None:
        paths_group = meta_group.create_group("results_paths")
        for k, v in results_paths.items():
            _write_h5_value(paths_group, k, v)


def get_solver_function(backend_name):
    backend_name = backend_name.lower().strip()
    if backend_name == "truss":
        return importlib.import_module("truss_solver").solve_truss
    elif backend_name == "responsegt":
        return importlib.import_module("ResGT_optimizer_adapter").solve_truss
    elif backend_name == "beam":
        return importlib.import_module("beam_optimizer_adapter").solve_truss
    else:
        raise ValueError(f"Unknown solver backend '{backend_name}'")


def _extract_translation_u(u: np.ndarray, n_nodes: int) -> np.ndarray:
    """Return (n_nodes,3) translations from either 3N (truss/ResGT) or 6N (beam) DOF vectors."""
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
      My = -(x_cp - elastic_axis_x) * Fz
    """
    f3 = distributed_vertical_load_to_nodes(nodes, wing.span, total_lift, distribution=distribution)
    if backend_name.lower().strip() != "beam":
        return f3

    n_nodes = nodes.shape[0]
    f6 = np.zeros(6 * n_nodes, dtype=float)
    for n in range(n_nodes):
        f6[6*n:6*n+3] = f3[3*n:3*n+3]

    if physics_settings.get("excite_torsion", False):
        elastic_axis_x = float(physics_settings.get("elastic_axis_x_frac", 0.25)) * float(wing.chord)
        x_cp = float(physics_settings.get("x_cp_frac", 0.35)) * float(wing.chord)
        dx = x_cp - elastic_axis_x
        for n in range(n_nodes):
            Fz = f6[6*n + 2]
            # r x F with r=(dx,0,0), F=(0,0,Fz) -> My = -dx * Fz
            f6[6*n + 4] += -dx * Fz
    return f6

def _build_node_adjacency(n_nodes, edges):
    nbrs = [set() for _ in range(n_nodes)]
    for i, j in edges:
        nbrs[int(i)].add(int(j))
        nbrs[int(j)].add(int(i))
    return nbrs


def _bfs_nodes_within_k_hops(seed, adjacency, k):
    visited = {int(seed)}
    frontier = {int(seed)}
    for _ in range(int(k)):
        new_frontier = set()
        for u in frontier:
            new_frontier.update(adjacency[u])
        new_frontier -= visited
        visited |= new_frontier
        frontier = new_frontier
        if not frontier:
            break
    return np.array(sorted(visited), dtype=int)


def _edge_coords_from_original(nodes_original, edges_original, edge_idx_original):
    edge_idx_original = np.asarray(edge_idx_original, dtype=int)
    if edge_idx_original.size == 0:
        return np.zeros((0, 2, 3), dtype=float)
    ee = edges_original[edge_idx_original]
    return np.stack([nodes_original[ee[:, 0]], nodes_original[ee[:, 1]]], axis=1)


def apply_local_edge_ring_damage(nodes, edges, areas, seed, k=0):
    adjacency = _build_node_adjacency(len(nodes), edges)
    impacted_nodes = _bfs_nodes_within_k_hops(seed, adjacency, k)
    impacted_set = set(int(v) for v in impacted_nodes)
    remove_edge_mask = np.array(
        [(int(i) in impacted_set) or (int(j) in impacted_set) for i, j in edges],
        dtype=bool,
    )
    removed_edge_idx_original = np.where(remove_edge_mask)[0]
    return {
        "seed": int(seed),
        "removed_nodes_original_idx": impacted_nodes,
        "removed_edges_original_idx": removed_edge_idx_original,
        "edges_kept_pre_filter": edges[~remove_edge_mask],
        "areas_kept_pre_filter": areas[~remove_edge_mask],
    }


def evaluate_damage_case(
    wing,
    nodes,
    edges,
    areas,
    seed,
    damage_mode="local_edge_ring",
    damage_k=0,
):
    if damage_mode == "legacy_graph_hops":
        info = damage_and_check_full_connectivity(
            wing=wing,
            nodes=nodes,
            edges=edges,
            areas=areas,
            seed=int(seed),
            k=int(damage_k),
        )
        removed_nodes_original_idx = np.asarray(
            info.get("removed_nodes_original_idx", np.zeros((0,), dtype=int)),
            dtype=int,
        )
        removed_node_set = set(removed_nodes_original_idx.tolist())
        removed_edge_mask = np.array(
            [(int(i) in removed_node_set) or (int(j) in removed_node_set) for i, j in edges],
            dtype=bool,
        )
        info["removed_edges_original_idx"] = np.where(removed_edge_mask)[0]
        info["damage_mode"] = damage_mode
        info["seed"] = int(seed)
        return info

    mild = apply_local_edge_ring_damage(nodes, edges, areas, seed=seed, k=damage_k)
    nodes_d, edges_d, areas_d, keep_nodes_intact, edge_keep_mask_intact = filter_to_root_connected_intact(
        nodes,
        mild["edges_kept_pre_filter"],
        mild["areas_kept_pre_filter"],
        wing,
    )
    connected = is_connected_safe(len(nodes_d), edges_d)
    graph_screen_passed = connected and (len(nodes_d) > 0) and (len(edges_d) > 0)
    return {
        "connected": connected,
        "graph_screen_passed": graph_screen_passed,
        "nodes_d": nodes_d,
        "edges_d": edges_d,
        "areas_d": areas_d,
        "removed_nodes_original_idx": mild["removed_nodes_original_idx"],
        "removed_edges_original_idx": mild["removed_edges_original_idx"],
        "keep_nodes_root_connected": keep_nodes_intact,
        "edge_keep_mask_root_connected": edge_keep_mask_intact,
        "damage_mode": damage_mode,
        "seed": int(seed),
    }


def try_grouped_prune_and_validate(
    solver,
    backend_name,
    physics_settings,
    wing,
    nodes,
    edges,
    areas,
    sigma_allow,
    res,
    E_modulus,
    total_lift,
    cond_max,
    damage_mode,
    damage_k,
):
    groups, labels = build_spanwise_bay_groups(nodes, edges, wing, n_span_bins=14)
    print(f"  grouped pruning: {len(groups)} candidate groups")

    if len(groups) == 0:
        return {
            "nodes_next": nodes,
            "edges_next": edges,
            "areas_next": areas,
            "removed_edges_idx": np.zeros((0,), dtype=int),
            "removed_edges_coords": np.zeros((0, 2, 3), dtype=float),
            "prune_ok": False,
            "prune_reason": "no_groups_available",
            "accepted_label": "",
        }

    scores = evaluate_group_scores(
        edges,
        areas,
        res.member_force,
        res.member_stress,
        groups,
        sigma_allow,
    )
    order = np.argsort(scores)

    for idx in order:
        gidx = groups[idx]
        glabel = labels[idx]

        trial_edges, trial_areas, removed_edges_idx = prune_one_group(edges, areas, gidx)
        removed_edges_coords = _edge_coords_from_original(nodes, edges, removed_edges_idx)

        trial_nodes, trial_edges, trial_areas, _, _ = filter_to_root_connected_intact(
            nodes, trial_edges, trial_areas, wing
        )

        if not is_connected_safe(len(trial_nodes), trial_edges):
            print(f"    reject {glabel}: intact disconnected")
            continue

        trial_root_mask = wing.root_mask(trial_nodes)
        trial_fixed_dofs = make_fixed_dofs_for_root_clamp(trial_nodes, trial_root_mask)

        try:
            f_trial = build_force_vector(
                trial_nodes, wing, total_lift, backend_name, physics_settings, distribution="elliptic"
            )
            trial_tip = choose_tip_node(trial_nodes, wing)
            trial_res = solver(
                trial_nodes,
                trial_edges,
                trial_areas,
                E_modulus,
                f_trial,
                trial_fixed_dofs,
                wing,
                regularization=0.0,
                tip_node_idx=trial_tip,
            )
            if (not np.isfinite(trial_res.cond_est)) or (trial_res.cond_est > cond_max):
                print(f"    reject {glabel}: intact mechanism/ill-conditioned (cond={trial_res.cond_est:.3e})")
                continue
        except Exception as err:
            print(f"    reject {glabel}: intact singular ({err})")
            continue

        target = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
        seed = int(np.argmin(np.linalg.norm(trial_nodes - target.reshape(1, 3), axis=1)))

        damage_info = evaluate_damage_case(
            wing=wing,
            nodes=trial_nodes,
            edges=trial_edges,
            areas=trial_areas,
            seed=seed,
            damage_mode=damage_mode,
            damage_k=damage_k,
        )

        if (not damage_info["connected"]) or (not damage_info.get("graph_screen_passed", False)):
            print(f"    reject {glabel}: damaged screen failed")
            continue

        trial_nodes_d = damage_info["nodes_d"]
        trial_edges_d = damage_info["edges_d"]
        trial_areas_d = damage_info["areas_d"]

        trial_root_mask_d = wing.root_mask(trial_nodes_d)
        trial_fixed_dofs_d = make_fixed_dofs_for_root_clamp(trial_nodes_d, trial_root_mask_d)

        try:
            f_trial_d = build_force_vector(
                trial_nodes_d, wing, total_lift, backend_name, physics_settings, distribution="elliptic"
            )
            trial_tip_d = choose_tip_node(trial_nodes_d, wing)
            trial_res_d = solver(
                trial_nodes_d,
                trial_edges_d,
                trial_areas_d,
                E_modulus,
                f_trial_d,
                trial_fixed_dofs_d,
                wing,
                regularization=0.0,
                tip_node_idx=trial_tip_d,
            )
            if (not np.isfinite(trial_res_d.cond_est)) or (trial_res_d.cond_est > cond_max):
                print(f"    reject {glabel}: damaged mechanism/ill-conditioned (cond={trial_res_d.cond_est:.3e})")
                continue
        except Exception as err:
            print(f"    reject {glabel}: damaged singular ({err})")
            continue

        print(f"    accept {glabel}: removed {len(removed_edges_idx)} edges")
        return {
            "nodes_next": trial_nodes,
            "edges_next": trial_edges,
            "areas_next": trial_areas,
            "removed_edges_idx": np.asarray(removed_edges_idx, dtype=int),
            "removed_edges_coords": removed_edges_coords,
            "prune_ok": True,
            "prune_reason": f"prune_accepted_{glabel}",
            "accepted_label": glabel,
        }

    return {
        "nodes_next": nodes,
        "edges_next": edges,
        "areas_next": areas,
        "removed_edges_idx": np.zeros((0,), dtype=int),
        "removed_edges_coords": np.zeros((0, 2, 3), dtype=float),
        "prune_ok": False,
        "prune_reason": "prune_rejected_all_groups",
        "accepted_label": "",
    }


def _set_equal_3d_axes(ax, xyz_min, xyz_max):
    xr, yr, zr = xyz_max[0] - xyz_min[0], xyz_max[1] - xyz_min[1], xyz_max[2] - xyz_min[2]
    r = 0.5 * max(xr, yr, zr, 1e-9)
    cx, cy, cz = 0.5 * (xyz_min[0] + xyz_max[0]), 0.5 * (xyz_min[1] + xyz_max[1]), 0.5 * (xyz_min[2] + xyz_max[2])
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def plot_edges_by_stress(ax, nodes, edges, stress, sigma_allow, cmap, norm, lw=1.1, alpha=0.95):
    if len(edges) == 0:
        return
    vals = np.abs(np.asarray(stress).reshape(-1)) / max(float(sigma_allow), 1e-16)
    vals = np.clip(vals, 0.0, 1.0)
    for k, (i, j) in enumerate(edges):
        p = nodes[int(i)]
        q = nodes[int(j)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=cmap(norm(vals[k])), linewidth=lw, alpha=alpha)


def plot_edges_background(ax, nodes, edges, color="0.75", lw=0.8, alpha=0.6, ls="--"):
    if len(edges) == 0:
        return
    for i, j in edges:
        p = nodes[int(i)]
        q = nodes[int(j)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=lw, alpha=alpha, linestyle=ls)


def plot_removed_edge_coords(ax, edge_coords, color="k", lw=2.8, alpha=1.0, ls="-"):
    if edge_coords is None or len(edge_coords) == 0:
        return
    for seg in edge_coords:
        p, q = seg
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=lw, alpha=alpha, linestyle=ls)


def scatter_nodes(ax, pts, color, marker, size, label=None, alpha=1.0):
    if pts is None or len(pts) == 0:
        return
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker=marker, s=size, label=label, alpha=alpha)


def plot_single_node(ax, coord, color, marker, size, label=None):
    if coord is None:
        return
    coord = np.asarray(coord).reshape(1, 3)
    scatter_nodes(ax, coord, color=color, marker=marker, size=size, label=label)


def make_movie_frame(
    out_png,
    out_svg,
    wing,
    it,
    sigma_allow,
    backend_name,
    intact_state,
    damaged_state,
    root_mask_intact,
    root_mask_damaged,
    removed_edges_prune_prev_coords,
    removed_edges_damage_coords,
    removed_damage_node_coords,
    info_text,
):
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    all_pts = [intact_state["nodes"]]
    if "nodes_def" in intact_state:
        all_pts.append(intact_state["nodes_def"])
    if damaged_state["nodes"].size > 0:
        all_pts.append(damaged_state["nodes"])
    if "nodes_def" in damaged_state and damaged_state["nodes_def"].size > 0:
        all_pts.append(damaged_state["nodes_def"])
    if removed_damage_node_coords.size > 0:
        all_pts.append(removed_damage_node_coords)
    xyz = np.vstack(all_pts)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)

    fig = plt.figure(figsize=(16, 8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 2.5], width_ratios=[1, 1])
    ax_i = fig.add_subplot(gs[0, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")

    fig.suptitle(f"Pruning + damage evolution ({backend_name})", fontsize=22, y=0.98)

    if settings["output"]["plot_undeformed_background"]:
        plot_edges_background(ax_i, intact_state["nodes"], intact_state["edges"], color="0.75", lw=0.8, alpha=0.6, ls="--")
    plot_edges_by_stress(
        ax_i,
        intact_state["nodes_def"] if settings["output"]["show_deformed_shape"] else intact_state["nodes"],
        intact_state["edges"],
        intact_state["member_stress"],
        sigma_allow,
        cmap,
        norm,
        lw=1.3,
        alpha=0.95,
    )
    plot_removed_edge_coords(ax_i, removed_edges_prune_prev_coords, color="k", lw=3.0, alpha=1.0)
    scatter_nodes(ax_i, intact_state["nodes"][root_mask_intact], color="#ff7f0e", marker="s", size=40, label="root clamp")
    plot_single_node(
        ax_i,
        intact_state["tip_coord_def"] if settings["output"]["show_deformed_shape"] else intact_state["tip_coord"],
        color="#2ca02c",
        marker="o",
        size=110,
        label="tip node",
    )
    plot_single_node(
        ax_i,
        intact_state["max_span_coord_def"] if settings["output"]["show_deformed_shape"] else intact_state["max_span_coord"],
        color="#d62728",
        marker="^",
        size=130,
        label="max span",
    )
    ax_i.set_title(f"Intact | iter {it:02d}")
    ax_i.set_xlabel("x")
    ax_i.set_ylabel("y")
    ax_i.set_zlabel("z")
    _set_equal_3d_axes(ax_i, xyz_min, xyz_max)
    ax_i.view_init(elev=18, azim=-62)
    ax_i.legend(loc="upper left", fontsize=11)

    if damaged_state["nodes"].size > 0 and damaged_state["edges"].size > 0:
        if settings["output"]["plot_undeformed_background"]:
            plot_edges_background(ax_d, damaged_state["nodes"], damaged_state["edges"], color="0.75", lw=0.8, alpha=0.6, ls="--")
        plot_edges_by_stress(
            ax_d,
            damaged_state["nodes_def"] if settings["output"]["show_deformed_shape"] else damaged_state["nodes"],
            damaged_state["edges"],
            damaged_state["member_stress"],
            sigma_allow,
            cmap,
            norm,
            lw=1.3,
            alpha=0.95,
        )
    plot_removed_edge_coords(ax_d, removed_edges_damage_coords, color="#9467bd", lw=2.6, alpha=0.95)
    plot_removed_edge_coords(ax_d, removed_edges_prune_prev_coords, color="k", lw=2.3, alpha=0.9, ls="--")
    if damaged_state["nodes"].size > 0:
        scatter_nodes(ax_d, damaged_state["nodes"][root_mask_damaged], color="#ff7f0e", marker="s", size=40, label="root clamp")
        plot_single_node(
            ax_d,
            damaged_state["tip_coord_def"] if settings["output"]["show_deformed_shape"] else damaged_state["tip_coord"],
            color="#2ca02c",
            marker="o",
            size=110,
            label="tip node",
        )
        plot_single_node(
            ax_d,
            damaged_state["max_span_coord_def"] if settings["output"]["show_deformed_shape"] else damaged_state["max_span_coord"],
            color="#d62728",
            marker="^",
            size=130,
            label="max span",
        )
    scatter_nodes(ax_d, removed_damage_node_coords, color="#9467bd", marker="x", size=100, label="removed damage nodes")
    ax_d.set_title(f"Damaged | iter {it:02d}")
    ax_d.set_xlabel("x")
    ax_d.set_ylabel("y")
    ax_d.set_zlabel("z")
    _set_equal_3d_axes(ax_d, xyz_min, xyz_max)
    ax_d.view_init(elev=18, azim=-62)
    ax_d.legend(loc="upper left", fontsize=11)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.20, 0.015, 0.62])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$|\sigma| / \sigma_{allow}$", fontsize=13)

    ax_txt.text(0.01, 0.98, info_text, va="top", ha="left", family="monospace", fontsize=14)

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def assemble_gif_and_mp4(frame_dir, gif_path, mp4_path, fps=2):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.lower().endswith(".png")])
    if not frame_files:
        return False, False

    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(gif_path, images, duration=1.0 / max(float(fps), 1.0))

    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h == 0 and pad_w == 0:
            padded_images.append(img)
            continue
        pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w))
        padded_images.append(np.pad(img, pad_width, mode="edge"))

    mp4_ok = True
    try:
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", macro_block_size=1, ffmpeg_log_level="warning") as writer:
            for img in padded_images:
                writer.append_data(img)
    except Exception as err:
        mp4_ok = False
        print(f"MP4 save failed: {err}")

    return True, mp4_ok


def save_history_plots(hist, sigma_allow, u_tip_max, summary_svg, summary_png):
    if len(hist["mass"]) == 0:
        return None, None

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    axs[0, 0].plot(hist["mass"], marker="o")
    axs[0, 0].set_title("Mass vs iteration")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Mass (kg)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(hist["sigma_intact"], marker="o", label="Intact")
    axs[0, 1].plot(hist["sigma_damaged"], marker="s", label="Damaged")
    axs[0, 1].axhline(sigma_allow / 1e6, color="r", linestyle="--", label="Allowable")
    axs[0, 1].set_title("Max stress vs iteration")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Stress (MPa)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(hist["tip_intact"], marker="o", label="Intact")
    axs[1, 0].plot(hist["tip_damaged"], marker="s", label="Damaged")
    axs[1, 0].axhline(u_tip_max, color="r", linestyle="--", label="Limit")
    axs[1, 0].set_title("Tip displacement vs iteration")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Tip displacement (m)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].bar(np.arange(len(hist["pruned_count_prev"])), hist["pruned_count_prev"])
    axs[1, 1].set_title("Edges pruned per iteration")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(summary_png, dpi=200, bbox_inches="tight")
    fig.savefig(summary_svg, bbox_inches="tight")
    plt.close(fig)
    return summary_svg, summary_png


def save_solver_comparison_plot(all_hists, sigma_allow, out_svg, out_png):
    backends = list(all_hists.keys())
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    for name in backends:
        hist = all_hists[name]
        axs[0, 0].plot(hist["mass"], marker="o", label=name)
        axs[0, 1].plot(hist["sigma_intact"], marker="o", label=f"{name} intact")
        axs[1, 0].plot(hist["tip_intact"], marker="o", label=f"{name} intact")
        axs[1, 1].plot(hist["tip_damaged"], marker="s", label=f"{name} damaged")

    axs[0, 0].set_title("Mass vs iteration")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Mass (kg)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].axhline(sigma_allow / 1e6, color="r", linestyle="--", label="Allowable")
    axs[0, 1].set_title("Intact max stress vs iteration")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Stress (MPa)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].set_title("Intact tip displacement vs iteration")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Tip displacement (m)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].set_title("Damaged tip displacement vs iteration")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Tip displacement (m)")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def run_backend(backend_name, settings, results_paths):
    solver = get_solver_function(backend_name)

    run_settings = settings["run"]
    solver_settings = settings["solvers"]
    wing_settings = settings["wing"]
    physics_settings = settings["physics"]
    optimization_settings = settings["optimization"]
    damage_settings = settings["damage"]
    lattice_settings = settings["lattice"]
    output_settings = settings["output"]

    wing = WingBox(
        span=wing_settings["span"],
        chord=wing_settings["chord"],
        depth=wing_settings["depth"],
        root_tol=wing_settings["root_tol"],
    )

    lattice_type = run_settings["lattice_type"]
    E_modulus = solver_settings["E_modulus"]
    density = solver_settings["density"]
    sigma_allow = solver_settings["sigma_allow"]
    u_tip_max = solver_settings["u_tip_max"]
    cond_max = solver_settings["cond_max"]
    a_min = optimization_settings["a_min"]
    a_max = optimization_settings["a_max"]
    a_init = optimization_settings["a_init"]
    total_lift = physics_settings["load_factor"] * wing_settings["m_vehicle"] * physics_settings["g"]
    n_iter = optimization_settings["max_iterations"]
    prune_after_iter = optimization_settings["prune_after_iter"]
    damage_mode = damage_settings["damage_mode"]
    damage_k = damage_settings["damage_k"]
    seed_span_fraction = damage_settings["seed_span_fraction"]
    seed_x_fraction = damage_settings["seed_x_fraction"]

    backend_root = os.path.join(results_paths["root"], backend_name)
    backend_data = os.path.join(backend_root, "data")
    backend_plots = os.path.join(backend_root, "plots")
    backend_frames = os.path.join(backend_root, "frames")
    backend_fem = os.path.join(backend_root, "fem")

    os.makedirs(backend_root, exist_ok=True)
    os.makedirs(backend_data, exist_ok=True)
    os.makedirs(backend_plots, exist_ok=True)
    os.makedirs(backend_frames, exist_ok=True)
    os.makedirs(backend_fem, exist_ok=True)

    h5_path = os.path.join(backend_data, f"lattice_opt_{backend_name}_{lattice_type}.h5")
    frame_dir = backend_frames
    gif_path = os.path.join(backend_plots, f"lattice_opt_{backend_name}_{lattice_type}.gif")
    mp4_path = os.path.join(backend_plots, f"lattice_opt_{backend_name}_{lattice_type}.mp4")
    summary_svg = os.path.join(backend_plots, f"lattice_opt_{backend_name}_{lattice_type}_summary.svg")
    summary_png = os.path.join(backend_plots, f"lattice_opt_{backend_name}_{lattice_type}_summary.png")

    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    nodes, edges = build_lattice(wing, lattice_type, lattice_settings)
    areas = np.full(len(edges), a_init, dtype=float)
    nodes, edges, areas, _, _ = filter_to_root_connected_intact(nodes, edges, areas, wing)

    print(f"\n=== Running backend: {backend_name} ===")
    print(f"  nodes={len(nodes)}, edges={len(edges)}, total_lift={total_lift:.2f} N")
    print(f"  HDF5={os.path.abspath(h5_path)}")

    hist = {
        "mass": [],
        "tip_intact": [],
        "tip_damaged": [],
        "sigma_intact": [],
        "sigma_damaged": [],
        "n_edges": [],
        "pruned_count_prev": [],
    }

    prev_prune_removed_edge_coords = np.zeros((0, 2, 3), dtype=float)
    prev_prune_label = "prune_not_attempted"

    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("meta")
        write_settings_meta(
            meta_group=meta,
            settings=settings,
            results_paths=results_paths,
            backend_name=backend_name,
            lattice_type=lattice_type,
        )

        for it in range(n_iter):
            intact_connected = is_connected_safe(len(nodes), edges)
            if not intact_connected:
                print(f"Iter {it:02d} | intact graph disconnected -> FAIL")
                break

            root_mask = wing.root_mask(nodes)
            fixed_dofs = make_fixed_dofs_for_root_clamp(nodes, root_mask)
            f = build_force_vector(nodes, wing, total_lift, backend_name, physics_settings, distribution="elliptic")
            tip_node_idx = choose_tip_node(nodes, wing)
            tip_coord = nodes[tip_node_idx]
            max_span_node_idx = int(np.argmax(nodes[:, 1]))
            max_span_coord = nodes[max_span_node_idx]

            try:
                res = solver(nodes, edges, areas, E_modulus, f, fixed_dofs, wing, regularization=0.0, tip_node_idx=tip_node_idx)
            except Exception as err:
                print(f"Iter {it:02d} | intact solve failed -> {err}")
                break

            if (not np.isfinite(res.cond_est)) or (res.cond_est > cond_max):
                print(f"Iter {it:02d} | intact cond too large ({res.cond_est:.3e}) -> FAIL")
                break

            mass = estimate_mass(nodes, edges, areas, density)
            max_sigma = float(np.max(np.abs(res.member_stress)))
            tip_disp = float(res.tip_disp)
            lambda2 = algebraic_connectivity_safe(len(nodes), edges)
            ebc_stats = edge_betweenness_stats(len(nodes), edges)

            u_xyz = _extract_translation_u(res.u, nodes.shape[0])
            nodes_def = nodes + settings["output"]["deformation_scale"] * u_xyz
            tip_coord_def = nodes_def[tip_node_idx]
            max_span_coord_def = nodes_def[max_span_node_idx]

            target = np.array([seed_x_fraction * wing.chord, seed_span_fraction * wing.span, 0.0])
            damage_seed_node_idx = int(np.argmin(np.linalg.norm(nodes - target.reshape(1, 3), axis=1)))
            damage_seed_coord = nodes[damage_seed_node_idx]

            damage_info = evaluate_damage_case(
                wing=wing,
                nodes=nodes,
                edges=edges,
                areas=areas,
                seed=damage_seed_node_idx,
                damage_mode=damage_mode,
                damage_k=damage_k,
            )
            damaged_connected = bool(damage_info["connected"])
            damaged_screen_passed = bool(damage_info.get("graph_screen_passed", False))
            damage_reason = "ok"

            nodes_d = np.zeros((0, 3), dtype=float)
            edges_d = np.zeros((0, 2), dtype=int)
            areas_d = np.zeros((0,), dtype=float)
            root_mask_d = np.zeros((0,), dtype=bool)
            res_d = None
            dam_tip_node_idx = -1
            dam_tip_coord = np.array([np.nan, np.nan, np.nan])
            dam_max_span_node_idx = -1
            dam_max_span_coord = np.array([np.nan, np.nan, np.nan])
            dam_tip = np.inf
            dam_sigma = np.inf
            dam_cond = np.inf
            dam_lambda2 = 0.0
            dam_ebc = {"max": np.inf, "mean": np.inf, "var": np.inf}
            nodes_d_def = np.zeros((0, 3), dtype=float)
            dam_tip_coord_def = dam_tip_coord.copy()
            dam_max_span_coord_def = dam_max_span_coord.copy()

            if (not damaged_connected) or (not damaged_screen_passed):
                damage_reason = "damaged_graph_disconnected_or_screen_failed"
            else:
                nodes_d = damage_info["nodes_d"]
                edges_d = damage_info["edges_d"]
                areas_d = damage_info["areas_d"]
                root_mask_d = wing.root_mask(nodes_d)
                fixed_dofs_d = make_fixed_dofs_for_root_clamp(nodes_d, root_mask_d)
                f_d = build_force_vector(nodes_d, wing, total_lift, backend_name, physics_settings, distribution="elliptic")
                dam_tip_node_idx = choose_tip_node(nodes_d, wing)
                dam_tip_coord = nodes_d[dam_tip_node_idx]
                dam_max_span_node_idx = int(np.argmax(nodes_d[:, 1]))
                dam_max_span_coord = nodes_d[dam_max_span_node_idx]

                try:
                    res_d = solver(nodes_d, edges_d, areas_d, E_modulus, f_d, fixed_dofs_d, wing, regularization=0.0, tip_node_idx=dam_tip_node_idx)
                    dam_cond = res_d.cond_est
                    dam_lambda2 = algebraic_connectivity_safe(len(nodes_d), edges_d)
                    dam_ebc = edge_betweenness_stats(len(nodes_d), edges_d)

                    if (not np.isfinite(dam_cond)) or (dam_cond > cond_max):
                        damage_reason = "damaged_solver_instability"
                    else:
                        dam_tip = float(res_d.tip_disp)
                        dam_sigma = float(np.max(np.abs(res_d.member_stress)))
                        u_d_xyz = _extract_translation_u(res_d.u, nodes_d.shape[0])
                        nodes_d_def = nodes_d + settings["output"]["deformation_scale"] * u_d_xyz
                        dam_tip_coord_def = nodes_d_def[dam_tip_node_idx]
                        dam_max_span_coord_def = nodes_d_def[dam_max_span_node_idx]
                except Exception as err:
                    damage_reason = f"damaged_solve_failed:{err}"

            removed_damage_node_idx = np.asarray(damage_info.get("removed_nodes_original_idx", np.zeros((0,), dtype=int)), dtype=int)
            removed_damage_node_coords = nodes[removed_damage_node_idx] if removed_damage_node_idx.size > 0 else np.zeros((0, 3), dtype=float)
            removed_damage_edge_idx = np.asarray(damage_info.get("removed_edges_original_idx", np.zeros((0,), dtype=int)), dtype=int)
            removed_damage_edge_coords = _edge_coords_from_original(nodes, edges, removed_damage_edge_idx)

            intact_util = max_sigma / sigma_allow
            damaged_util = dam_sigma / sigma_allow if np.isfinite(dam_sigma) else np.inf

            info_text = (
                f"backend={backend_name}\n"
                f"iter={it:02d}\n"
                f"edges={len(edges):3d}  pruned_prev={len(prev_prune_removed_edge_coords):1d}\n"
                f"mass={mass:7.3f} kg\n"
                f"sigma_i={max_sigma/1e6:7.3f} MPa   sigma_d={dam_sigma/1e6 if np.isfinite(dam_sigma) else np.inf:7.3f} MPa\n"
                f"u_tip_i={tip_disp:8.4e} m   u_tip_d={dam_tip:8.4e} m\n"
                f"tip_y_i={tip_coord[1]:5.3f}   ymax_i={max_span_coord[1]:5.3f}\n"
                f"tip_y_d={dam_tip_coord[1] if np.isfinite(dam_tip_coord[1]) else np.nan:5.3f}   ymax_d={dam_max_span_coord[1] if np.isfinite(dam_max_span_coord[1]) else np.nan:5.3f}\n"
                f"damage_mode={damage_mode}  damage_k={damage_k}\n"
                f"torsion={'on' if physics_settings.get('excite_torsion', False) else 'off'}  ea={physics_settings.get('elastic_axis_x_frac', 0.25):.2f}c  cp={physics_settings.get('x_cp_frac', 0.35):.2f}c\n"
                f"damage_reason={damage_reason}\n"
                f"prev_prune={prev_prune_label}\n"
                f"deformation_scale={settings['output']['deformation_scale']:g}x"
            )

            frame_png = os.path.join(frame_dir, f"frame_{it:04d}.png")
            frame_svg = os.path.join(frame_dir, f"frame_{it:04d}.svg")

            intact_state = {
                "nodes": nodes,
                "nodes_def": nodes_def,
                "edges": edges,
                "member_stress": res.member_stress,
                "tip_coord": tip_coord,
                "tip_coord_def": tip_coord_def,
                "max_span_coord": max_span_coord,
                "max_span_coord_def": max_span_coord_def,
            }
            damaged_state = {
                "nodes": nodes_d,
                "nodes_def": nodes_d_def,
                "edges": edges_d,
                "member_stress": res_d.member_stress if res_d is not None and np.isfinite(dam_sigma) else np.zeros((len(edges_d),), dtype=float),
                "tip_coord": dam_tip_coord,
                "tip_coord_def": dam_tip_coord_def,
                "max_span_coord": dam_max_span_coord,
                "max_span_coord_def": dam_max_span_coord_def,
            }

            if output_settings["save_frames"]:
                make_movie_frame(
                    frame_png,
                    frame_svg,
                    wing,
                    it,
                    sigma_allow,
                    backend_name,
                    intact_state,
                    damaged_state,
                    root_mask,
                    root_mask_d,
                    prev_prune_removed_edge_coords,
                    removed_damage_edge_coords,
                    removed_damage_node_coords,
                    info_text,
                )

            save_iteration_hdf5(h5, it, {
                "solver_backend": backend_name,
                "nodes": nodes,
                "edges": edges,
                "areas": areas,
                "force_vector": f,
                "displacements": res.u,
                "member_force": res.member_force,
                "member_stress": res.member_stress,
                "mass": mass,
                "tip_disp": tip_disp,
                "max_sigma": max_sigma,
                "lambda2": lambda2,
                "cond_est": res.cond_est,
                "intact_connected": intact_connected,
                "ebc_stats": ebc_stats,
                "tip_node_idx": tip_node_idx,
                "tip_coord": tip_coord,
                "tip_y": float(tip_coord[1]),
                "max_span_node_idx": max_span_node_idx,
                "max_span_coord": max_span_coord,
                "max_span_y": float(max_span_coord[1]),
                "damage_reason": damage_reason,
                "damage_mode": damage_mode,
                "damage_k": int(damage_k),
                "damage_seed_node_idx": int(damage_seed_node_idx),
                "damage_seed_coord": damage_seed_coord,
                "damaged_nodes": nodes_d,
                "damaged_edges": edges_d,
                "damaged_areas": areas_d,
                "damaged_tip_disp": dam_tip,
                "damaged_max_sigma": dam_sigma,
                "damaged_lambda2": dam_lambda2,
                "damaged_cond_est": dam_cond,
                "damaged_connected": damaged_connected,
                "damaged_graph_screen_passed": damaged_screen_passed,
                "damaged_ebc_stats": dam_ebc,
                "damaged_tip_node_idx": int(dam_tip_node_idx),
                "damaged_tip_coord": dam_tip_coord,
                "damaged_max_span_node_idx": int(dam_max_span_node_idx),
                "damaged_max_span_coord": dam_max_span_coord,
                "removed_nodes_damage_original_idx": removed_damage_node_idx,
                "removed_edges_damage_original_idx": removed_damage_edge_idx,
                "removed_edges_prune_prev_coords": prev_prune_removed_edge_coords,
                "prev_prune_label": prev_prune_label,
                "frame_png": frame_png,
                "frame_svg": frame_svg,
            })

            print(
                f"Iter {it:02d} | backend={backend_name:10s} | edges={len(edges):5d} | mass={mass:8.2f} kg | "
                f"σ_intact={max_sigma/1e6:7.2f} MPa ({intact_util:5.3f}) | "
                f"σ_dmg={dam_sigma/1e6 if np.isfinite(dam_sigma) else np.inf:7.2f} MPa ({damaged_util:5.3f}) | "
                f"u_tip={tip_disp:8.4e} m | u_tip_d={dam_tip:8.4e} m | "
                f"cond_i={res.cond_est:.3e} cond_d={dam_cond:.3e} | {damage_reason}"
            )

            hist["mass"].append(mass)
            hist["tip_intact"].append(tip_disp)
            hist["tip_damaged"].append(dam_tip)
            hist["sigma_intact"].append(max_sigma / 1e6)
            hist["sigma_damaged"].append(dam_sigma / 1e6 if np.isfinite(dam_sigma) else np.nan)
            hist["n_edges"].append(len(edges))
            hist["pruned_count_prev"].append(len(prev_prune_removed_edge_coords))

            if optimization_settings.get("enable_area_sizing", True):
                areas_after_sizing = simple_sizing_update(
                    areas,
                    res.member_stress,
                    sigma_allow,
                    a_min,
                    a_max,
                    shrink_factor=optimization_settings["shrink_factor"],
                    grow_factor=optimization_settings["grow_factor"],
                )
            else:
                areas_after_sizing = areas.copy()

            if it >= prune_after_iter:
                prune_out = try_grouped_prune_and_validate(
                    solver=solver,
                    backend_name=backend_name,
                    physics_settings=physics_settings,
                    wing=wing,
                    nodes=nodes,
                    edges=edges,
                    areas=areas_after_sizing,
                    sigma_allow=sigma_allow,
                    res=res,
                    E_modulus=E_modulus,
                    total_lift=total_lift,
                    cond_max=cond_max,
                    damage_mode=damage_mode,
                    damage_k=damage_k,
                )
            else:
                prune_out = {
                    "nodes_next": nodes,
                    "edges_next": edges,
                    "areas_next": areas_after_sizing,
                    "removed_edges_idx": np.zeros((0,), dtype=int),
                    "removed_edges_coords": np.zeros((0, 2, 3), dtype=float),
                    "prune_ok": False,
                    "prune_reason": "prune_not_attempted",
                    "accepted_label": "",
                }

            prev_prune_removed_edge_coords = prune_out["removed_edges_coords"]
            prev_prune_label = prune_out["prune_reason"]
            nodes = prune_out["nodes_next"]
            edges = prune_out["edges_next"]
            areas = prune_out["areas_next"]

    gif_ok, mp4_ok = assemble_gif_and_mp4(frame_dir, gif_path, mp4_path, fps=output_settings["fps"])
    summary_svg, summary_png = save_history_plots(hist, sigma_allow, u_tip_max, summary_svg, summary_png)
    
    # ==============================================================
    # Export NASTRAN FEM for final lattice
    # ==============================================================

    if settings["output"].get("export_final_fem", False):

        try:

            fem_path = os.path.join(
                backend_fem,
                f"lattice_final_{backend_name}_{lattice_type}.bdf"
            )
            
            export_lattice_to_nastran(
                h5_file=h5_path,
                output_bdf=fem_path,
                iteration=it,
                target_element_length=settings["output"].get("fem_element_length", 0.03),
                write_active_loads=settings["output"].get("fem_include_loads", False),
                write_active_moments=settings["output"].get("fem_include_loads", False),
            )

            print(f"Final FEM exported: {fem_path}")

        except Exception as err:
            print("FEM export failed:", err)

    print(f"Saved debug data to HDF5: {os.path.abspath(h5_path)}")
    if gif_ok:
        print(f"Saved GIF: {os.path.abspath(gif_path)}")
    if mp4_ok:
        print(f"Saved MP4: {os.path.abspath(mp4_path)}")
    if summary_svg is not None:
        print(f"Saved summary SVG: {os.path.abspath(summary_svg)}")

    return hist, {
        "h5": h5_path,
        "gif": gif_path,
        "mp4": mp4_path,
        "summary_svg": summary_svg,
        "summary_png": summary_png,
    }


def main():
    base_results_dir = settings["output"]["results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    run_name = f"{settings['run']['run_name']}_{settings['run']['lattice_type']}"
    results_paths = make_results_paths(base_results_dir, run_name)

    backends = settings["run"]["backends"]
    all_hists = {}
    for backend_name in backends:
        hist, _ = run_backend(backend_name, settings, results_paths)
        all_hists[backend_name] = hist

    if len(backends) > 1:
        comp_svg = os.path.join(results_paths["plots"], f"solver_comparison_{settings['run']['lattice_type']}.svg")
        comp_png = os.path.join(results_paths["plots"], f"solver_comparison_{settings['run']['lattice_type']}.png")
        save_solver_comparison_plot(all_hists, settings["solvers"]["sigma_allow"], comp_svg, comp_png)
        print(f"Saved solver comparison SVG: {os.path.abspath(comp_svg)}")
        print(f"Saved solver comparison PNG: {os.path.abspath(comp_png)}")


if __name__ == "__main__":
    main()
