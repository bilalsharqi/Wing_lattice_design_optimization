import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py

from wingbox_domain import WingBox
from load_mapping import distributed_vertical_load_to_nodes
from truss_solver import solve_truss, choose_tip_node
from damage_models import (
    filter_to_root_connected_intact,
    damage_and_check_full_connectivity,
)
from gt_metrics import (
    is_connected_safe,
    algebraic_connectivity_safe,
    edge_betweenness_stats,
)

# Keep both generators available
from generate_octet_lattice import generate_octet_ground_structure
from generate_square_lattice import generate_square_wingbox_lattice

# Square-lattice-specific visualization
from viz_square_lattice import visualize_square_lattice, visualize_damage_on_square_lattice


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


def prune_members(
    edges,
    areas,
    stress,
    force,
    protect_root_nodes,
    sigma_allow,
    area_factor=1.10,
    stress_frac=0.03,
    force_frac=0.03,
):
    keep = np.ones(len(edges), dtype=bool)
    max_abs_force = max(float(np.max(np.abs(force))), 1e-16)

    low_area = areas <= area_factor * np.min(areas)
    low_stress = np.abs(stress) <= stress_frac * sigma_allow
    low_force = np.abs(force) <= force_frac * max_abs_force

    removed_idx = []
    for e, (i, j) in enumerate(edges):
        if i in protect_root_nodes or j in protect_root_nodes:
            continue
        if low_area[e] and low_stress[e] and low_force[e]:
            keep[e] = False
            removed_idx.append(e)

    if np.sum(keep) == 0:
        keep[:] = True
        removed_idx = []

    return edges[keep], areas[keep], keep, np.array(removed_idx, dtype=int)


def try_prune_and_validate(
    wing,
    nodes,
    edges,
    areas,
    root_nodes,
    sigma_allow,
    res,
    E_modulus,
    total_lift,
    cond_max,
    damage_k,
):
    """
    Try a pruning step. Accept only if BOTH:
      1) intact trial graph remains connected and solvable
      2) damaged trial graph remains connected and solvable
    """
    trial_edges, trial_areas, keep_mask, removed_edges_idx = prune_members(
        edges,
        areas,
        res.member_stress,
        res.member_force,
        root_nodes,
        sigma_allow,
        area_factor=1.10,
        stress_frac=0.03,
        force_frac=0.03,
    )

    # Filter to intact root-connected component
    trial_nodes, trial_edges, trial_areas, keep_nodes_intact, edge_keep_mask_intact = filter_to_root_connected_intact(
        nodes,
        trial_edges,
        trial_areas,
        wing,
    )

    if not is_connected_safe(len(trial_nodes), trial_edges):
        return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_intact_disconnected"

    trial_root_mask = wing.root_mask(trial_nodes)
    trial_fixed_dofs = make_fixed_dofs_for_root_clamp(trial_nodes, trial_root_mask)

    try:
        f_trial = distributed_vertical_load_to_nodes(
            trial_nodes, wing.span, total_lift, distribution="elliptic"
        )
        trial_tip_node_idx = choose_tip_node(trial_nodes, wing)
        trial_res = solve_truss(
            trial_nodes,
            trial_edges,
            trial_areas,
            E_modulus,
            f_trial,
            trial_fixed_dofs,
            wing,
            regularization=0.0,
            tip_node_idx=trial_tip_node_idx,
        )

        if (not np.isfinite(trial_res.cond_est)) or (trial_res.cond_est > cond_max):
            return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_intact_mechanism"

    except Exception:
        return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_intact_singular"

    # Damaged trial check
    target = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
    seed = int(np.argmin(np.linalg.norm(trial_nodes - target.reshape(1, 3), axis=1)))

    damage_info = damage_and_check_full_connectivity(
        wing=wing,
        nodes=trial_nodes,
        edges=trial_edges,
        areas=trial_areas,
        seed=seed,
        k=damage_k,
    )

    if (not damage_info["connected"]) or (not damage_info.get("graph_screen_passed", False)):
        return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_damaged_screen"

    trial_nodes_d = damage_info["nodes_d"]
    trial_edges_d = damage_info["edges_d"]
    trial_areas_d = damage_info["areas_d"]

    trial_root_mask_d = wing.root_mask(trial_nodes_d)
    trial_fixed_dofs_d = make_fixed_dofs_for_root_clamp(trial_nodes_d, trial_root_mask_d)

    try:
        f_trial_d = distributed_vertical_load_to_nodes(
            trial_nodes_d, wing.span, total_lift, distribution="elliptic"
        )
        trial_tip_node_idx_d = choose_tip_node(trial_nodes_d, wing)
        trial_res_d = solve_truss(
            trial_nodes_d,
            trial_edges_d,
            trial_areas_d,
            E_modulus,
            f_trial_d,
            trial_fixed_dofs_d,
            wing,
            regularization=0.0,
            tip_node_idx=trial_tip_node_idx_d,
        )

        if (not np.isfinite(trial_res_d.cond_est)) or (trial_res_d.cond_est > cond_max):
            return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_damaged_mechanism"

    except Exception:
        return nodes, edges, areas, np.zeros((0,), dtype=int), False, "prune_rejected_damaged_singular"

    return trial_nodes, trial_edges, trial_areas, removed_edges_idx, True, "prune_accepted"


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
        else:
            grp.attrs[key] = json.dumps(val)


def plot_lattice(ax, nodes, edges, title, stride=1, color="C0", alpha=0.8):
    ax.set_title(title)
    if len(edges) == 0:
        return
    use = np.arange(0, len(edges), max(1, stride))
    for e in use:
        i, j = edges[e]
        p = nodes[i]
        q = nodes[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=0.8, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def build_lattice(wing, lattice_type):
    """
    Switch between lattice families here.
    """
    if lattice_type == "square":
        lat = generate_square_wingbox_lattice(
            span=wing.span,
            chord=wing.chord,
            depth=wing.depth,
            ny=14,
            nx=3,
            nz=2,
            add_xy_diagonals=True,
            add_yz_diagonals=True,
            add_xz_diagonals=True,
        )
    elif lattice_type == "octet":
        lat = generate_octet_ground_structure(
            wing.span,
            wing.chord,
            wing.depth,
            ny=21,
            nx=7,
            nz=5,
        )
    else:
        raise ValueError(f"Unknown lattice_type='{lattice_type}'")

    return lat.nodes, lat.edges


def main():
    wing = WingBox(span=4.0, chord=1.0, depth=0.16, root_tol=1e-9)

    # -----------------------------
    # USER CHOICE: switch graph here
    # -----------------------------
    lattice_type = "square"
    # lattice_type = "octet"

    E_modulus = 70e9
    density = 2700.0
    sigma_allow = 250e6
    u_tip_max = 0.25
    cond_max = 1e10

    a_min = 5e-7
    a_max = 5e-4
    a_init = 5e-5

    g = 9.80665
    m_vehicle = 50.0
    load_factor = 2.5
    total_lift = load_factor * m_vehicle * g

    n_iter = 25
    prune_after_iter = 1
    damage_k = 1

    h5_path = f"lattice_opt_debug_v4_{lattice_type}.h5"

    nodes, edges = build_lattice(wing, lattice_type)
    areas = np.full(len(edges), a_init, dtype=float)

    # enforce root-connected intact component from the start
    nodes, edges, areas, keep_nodes_intact, _ = filter_to_root_connected_intact(
        nodes, edges, areas, wing
    )

    root_mask = wing.root_mask(nodes)
    root_nodes = np.where(root_mask)[0]
    fixed_dofs = make_fixed_dofs_for_root_clamp(nodes, root_mask)

    print("Initial design summary:")
    print(f"  lattice type   = {lattice_type}")
    print(f"  nodes          = {len(nodes)}")
    print(f"  edges          = {len(edges)}")
    print(f"  total lift     = {total_lift:.2f} N")
    print(f"  HDF5 output    = {os.path.abspath(h5_path)}")

    # Visualization of starting design
    if lattice_type == "square":
        visualize_square_lattice(nodes, edges, wing)

    hist = {
        "mass": [],
        "tip_intact": [],
        "tip_damaged": [],
        "sigma_intact": [],
        "sigma_damaged": [],
        "n_edges": [],
        "lambda2_intact": [],
        "lambda2_damaged": [],
        "pruned_count": [],
        "intact_connected": [],
        "damaged_connected": [],
    }

    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["lattice_type"] = lattice_type
        meta.attrs["span"] = wing.span
        meta.attrs["chord"] = wing.chord
        meta.attrs["depth"] = wing.depth
        meta.attrs["E_modulus"] = E_modulus
        meta.attrs["density"] = density
        meta.attrs["sigma_allow"] = sigma_allow
        meta.attrs["u_tip_max"] = u_tip_max
        meta.attrs["load_factor"] = load_factor
        meta.attrs["m_vehicle"] = m_vehicle
        meta.attrs["total_lift"] = total_lift
        meta.attrs["cond_max"] = cond_max

        for it in range(n_iter):
            intact_connected = is_connected_safe(len(nodes), edges)

            if not intact_connected:
                print(f"Iter {it:02d} | intact graph disconnected -> FAIL")
                break

            root_mask = wing.root_mask(nodes)
            root_nodes = np.where(root_mask)[0]
            fixed_dofs = make_fixed_dofs_for_root_clamp(nodes, root_mask)

            f = distributed_vertical_load_to_nodes(
                nodes, wing.span, total_lift, distribution="elliptic"
            )
            tip_node_idx = choose_tip_node(nodes, wing)

            try:
                res = solve_truss(
                    nodes,
                    edges,
                    areas,
                    E_modulus,
                    f,
                    fixed_dofs,
                    wing,
                    regularization=0.0,
                    tip_node_idx=tip_node_idx,
                )
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

            target = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
            seed = int(np.argmin(np.linalg.norm(nodes - target.reshape(1, 3), axis=1)))

            damage_info = damage_and_check_full_connectivity(
                wing=wing,
                nodes=nodes,
                edges=edges,
                areas=areas,
                seed=seed,
                k=damage_k,
            )

            damaged_connected = damage_info["connected"]
            damaged_screen_passed = damage_info.get("graph_screen_passed", False)
            damage_reason = "ok"

            if (not damaged_connected) or (not damaged_screen_passed):
                dam_tip = np.inf
                dam_sigma = np.inf
                dam_lambda2 = 0.0
                dam_ebc = {"max": np.inf, "mean": np.inf, "var": np.inf}
                dam_tip_node_idx = -1
                dam_cond = np.inf
                damage_reason = "damaged_graph_disconnected_or_screen_failed"
            else:
                nodes_d = damage_info["nodes_d"]
                edges_d = damage_info["edges_d"]
                areas_d = damage_info["areas_d"]

                root_mask_d = wing.root_mask(nodes_d)
                fixed_dofs_d = make_fixed_dofs_for_root_clamp(nodes_d, root_mask_d)
                f_d = distributed_vertical_load_to_nodes(
                    nodes_d, wing.span, total_lift, distribution="elliptic"
                )
                dam_tip_node_idx = choose_tip_node(nodes_d, wing)

                try:
                    res_d = solve_truss(
                        nodes_d,
                        edges_d,
                        areas_d,
                        E_modulus,
                        f_d,
                        fixed_dofs_d,
                        wing,
                        regularization=0.0,
                        tip_node_idx=dam_tip_node_idx,
                    )
                    dam_cond = res_d.cond_est
                    if (not np.isfinite(dam_cond)) or (dam_cond > cond_max):
                        dam_tip = np.inf
                        dam_sigma = np.inf
                        dam_lambda2 = algebraic_connectivity_safe(len(nodes_d), edges_d)
                        dam_ebc = edge_betweenness_stats(len(nodes_d), edges_d)
                        damage_reason = "damaged_solver_instability"
                    else:
                        dam_tip = float(res_d.tip_disp)
                        dam_sigma = float(np.max(np.abs(res_d.member_stress)))
                        dam_lambda2 = algebraic_connectivity_safe(len(nodes_d), edges_d)
                        dam_ebc = edge_betweenness_stats(len(nodes_d), edges_d)
                except Exception as err:
                    dam_tip = np.inf
                    dam_sigma = np.inf
                    dam_lambda2 = 0.0
                    dam_ebc = {"max": np.inf, "mean": np.inf, "var": np.inf}
                    dam_cond = np.inf
                    damage_reason = f"damaged_solve_failed:{err}"

            intact_util = max_sigma / sigma_allow
            damaged_util = dam_sigma / sigma_allow if np.isfinite(dam_sigma) else np.inf

            if not np.isfinite(dam_cond) or not np.isfinite(dam_tip) or not np.isfinite(dam_sigma):
                print(f"Iter {it:02d} | baseline design infeasible under damage -> stopping optimization")
                print(
                    f"Iter {it:02d} | edges={len(edges):5d} | mass={mass:8.2f} kg | "
                    f"σ_intact={max_sigma/1e6:7.2f} MPa ({intact_util:5.3f}) | "
                    f"σ_dmg=inf | u_tip={tip_disp:8.4e} m | u_tip_d=inf | "
                    f"conn_i={intact_connected} conn_d={damaged_connected} | "
                    f"cond_i={res.cond_est:.3e} cond_d=inf | {damage_reason}"
                )
                break

            # sizing update first
            areas = simple_sizing_update(
                areas, res.member_stress, sigma_allow, a_min, a_max
            )

            edges_before = len(edges)

            if it >= prune_after_iter:
                nodes, edges, areas, removed_edges_idx, prune_ok, prune_reason = try_prune_and_validate(
                    wing=wing,
                    nodes=nodes,
                    edges=edges,
                    areas=areas,
                    root_nodes=root_nodes,
                    sigma_allow=sigma_allow,
                    res=res,
                    E_modulus=E_modulus,
                    total_lift=total_lift,
                    cond_max=cond_max,
                    damage_k=damage_k,
                )
            else:
                removed_edges_idx = np.zeros((0,), dtype=int)
                prune_ok = False
                prune_reason = "prune_not_attempted"

            pruned_now = len(removed_edges_idx)

            print(
                f"Iter {it:02d} | edges={len(edges):5d} | mass={mass:8.2f} kg | "
                f"σ_intact={max_sigma/1e6:7.2f} MPa ({intact_util:5.3f}) | "
                f"σ_dmg={dam_sigma/1e6:7.2f} MPa ({damaged_util:5.3f}) | "
                f"u_tip={tip_disp:8.4e} m | u_tip_d={dam_tip:8.4e} m | "
                f"conn_i={intact_connected} conn_d={damaged_connected} | "
                f"cond_i={res.cond_est:.3e} cond_d={dam_cond:.3e} | "
                f"{damage_reason} | prune={prune_reason}"
            )

            save_iteration_hdf5(
                h5,
                it,
                {
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
                    "tip_node_idx": tip_node_idx,
                    "cond_est": res.cond_est,
                    "intact_connected": intact_connected,
                    "ebc_stats": ebc_stats,
                    "damage_reason": damage_reason,
                    "damaged_nodes": damage_info["nodes_d"] if damaged_connected else np.zeros((0, 3)),
                    "damaged_edges": damage_info["edges_d"] if damaged_connected else np.zeros((0, 2), dtype=int),
                    "damaged_areas": damage_info["areas_d"] if damaged_connected else np.zeros((0,)),
                    "removed_nodes_damage_original_idx": damage_info["removed_nodes_original_idx"],
                    "damaged_tip_disp": dam_tip,
                    "damaged_max_sigma": dam_sigma,
                    "damaged_lambda2": dam_lambda2,
                    "damaged_tip_node_idx": dam_tip_node_idx,
                    "damaged_cond_est": dam_cond,
                    "damaged_connected": damaged_connected,
                    "damaged_graph_screen_passed": damaged_screen_passed,
                    "damaged_ebc_stats": dam_ebc,
                    "removed_edges_idx": removed_edges_idx,
                    "pruned_count": pruned_now,
                    "prune_reason": prune_reason,
                },
            )

            hist["mass"].append(mass)
            hist["tip_intact"].append(tip_disp)
            hist["tip_damaged"].append(dam_tip)
            hist["sigma_intact"].append(max_sigma / 1e6)
            hist["sigma_damaged"].append(dam_sigma / 1e6)
            hist["n_edges"].append(edges_before)
            hist["lambda2_intact"].append(lambda2)
            hist["lambda2_damaged"].append(dam_lambda2)
            hist["pruned_count"].append(pruned_now)
            hist["intact_connected"].append(intact_connected)
            hist["damaged_connected"].append(damaged_connected)

    if len(hist["mass"]) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        axs[0, 0].plot(hist["mass"], marker="o")
        axs[0, 0].set_title("Mass vs iteration")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Mass (kg)")

        axs[0, 1].plot(hist["sigma_intact"], marker="o", label="Intact")
        axs[0, 1].plot(hist["sigma_damaged"], marker="s", label="Damaged")
        axs[0, 1].axhline(sigma_allow / 1e6, color="r", linestyle="--", label="Allowable")
        axs[0, 1].set_title("Max stress vs iteration")
        axs[0, 1].legend()

        axs[1, 0].plot(hist["tip_intact"], marker="o", label="Intact")
        axs[1, 0].plot(hist["tip_damaged"], marker="s", label="Damaged")
        axs[1, 0].axhline(u_tip_max, color="r", linestyle="--", label="Limit")
        axs[1, 0].set_title("Tip displacement vs iteration")
        axs[1, 0].legend()

        axs[1, 1].bar(np.arange(len(hist["pruned_count"])), hist["pruned_count"])
        axs[1, 1].set_title("Edges pruned per iteration")

        plt.tight_layout()
        plt.show()

    # optional damage visualization for square lattice
    if lattice_type == "square" and len(nodes) > 0 and len(edges) > 0:
        target = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
        seed = int(np.argmin(np.linalg.norm(nodes - target.reshape(1, 3), axis=1)))
        damage_info = damage_and_check_full_connectivity(
            wing=wing,
            nodes=nodes,
            edges=edges,
            areas=areas,
            seed=seed,
            k=damage_k,
        )
        visualize_damage_on_square_lattice(
            nodes,
            edges,
            damage_info["removed_nodes_original_idx"],
            wing,
        )

    print(f"\nSaved debug data to HDF5: {os.path.abspath(h5_path)}")


if __name__ == "__main__":
    main()