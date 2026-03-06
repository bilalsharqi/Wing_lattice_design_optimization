
import numpy as np
import matplotlib.pyplot as plt

from wingbox_domain import WingBox
from generate_octet_lattice import generate_octet_ground_structure
from load_mapping import distributed_vertical_load_to_nodes
from truss_solver import solve_truss
from damage_models import damage_and_keep_root_component
from gt_metrics import algebraic_connectivity_safe, edge_betweenness_stats


def make_fixed_dofs_for_root_clamp(nodes: np.ndarray, root_mask: np.ndarray) -> np.ndarray:
    fixed_nodes = np.where(root_mask)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes + 0, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    return np.unique(fixed_dofs)


def estimate_mass(nodes: np.ndarray, edges: np.ndarray, areas: np.ndarray, density: float) -> float:
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    L = np.linalg.norm(q - p, axis=1)
    return float(np.sum(density * L * areas))


def simple_sizing_update(
    areas: np.ndarray,
    stress: np.ndarray,
    sigma_allow: float,
    a_min: float,
    a_max: float,
    shrink_factor: float = 0.985,
    grow_factor: float = 1.06,
):
    """
    Mild sizing update:
      - grow violating members
      - shrink members far below allowable
    """
    new_a = areas.copy()
    ratio = np.abs(stress) / max(sigma_allow, 1e-16)
    new_a[ratio > 1.0] *= grow_factor
    new_a[ratio < 0.10] *= shrink_factor
    return np.clip(new_a, a_min, a_max)


def prune_members(
    edges: np.ndarray,
    areas: np.ndarray,
    stress: np.ndarray,
    area_threshold: float,
    stress_fraction_threshold: float,
    sigma_allow: float,
    protect_root_nodes: np.ndarray,
):
    """
    Remove members that are both:
      - near minimum area
      - very lightly stressed
    Keep members attached to root nodes.
    """
    keep = np.ones(len(edges), dtype=bool)

    low_area = areas <= area_threshold
    low_stress = np.abs(stress) <= (stress_fraction_threshold * sigma_allow)

    for e, (i, j) in enumerate(edges):
        if i in protect_root_nodes or j in protect_root_nodes:
            continue
        if low_area[e] and low_stress[e]:
            keep[e] = False

    if np.sum(keep) == 0:
        keep[:] = True

    return edges[keep], areas[keep], keep


def evaluate_damage_case(
    wing: WingBox,
    nodes: np.ndarray,
    edges: np.ndarray,
    areas: np.ndarray,
    E_modulus: float,
    total_lift: float,
    damage_mode: str = "khop",
    khop_seed: int = None,
    khop_k: int = 2,
    nearest_center: np.ndarray = None,
    nearest_n: int = 20,
):
    """
    Damage evaluation with stability fixes:
      1) apply damage
      2) keep only root-connected component
      3) if no root-connected structure remains, mark as failure
      4) solve with small regularization
    """
    if damage_mode == "khop":
        if khop_seed is None:
            target = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
            khop_seed = int(np.argmin(np.linalg.norm(nodes - target.reshape(1, 3), axis=1)))
        nodes_d, edges_d, keep_nodes, edge_keep_mask = damage_and_keep_root_component(
            wing=wing, nodes=nodes, edges=edges, mode="khop", seed=khop_seed, k=khop_k
        )
    elif damage_mode == "nearest_n":
        if nearest_center is None:
            nearest_center = np.array([0.5 * wing.chord, 0.6 * wing.span, 0.0])
        nodes_d, edges_d, keep_nodes, edge_keep_mask = damage_and_keep_root_component(
            wing=wing, nodes=nodes, edges=edges, mode="nearest_n", center_xyz=nearest_center, n_remove=nearest_n
        )
    else:
        raise ValueError(f"Unknown damage_mode='{damage_mode}'")

    # If the damaged structure is gone or has no root-connected component, treat as failed design
    if len(edges_d) == 0 or len(nodes_d) < 2 or np.sum(edge_keep_mask) == 0:
        return {
            "success": False,
            "reason": "no_root_connected_structure",
            "tip_disp": np.inf,
            "max_sigma": np.inf,
            "lambda2": 0.0,
            "ebc_max": np.inf,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
        }

    areas_d = areas[edge_keep_mask]

    root_mask_d = wing.root_mask(nodes_d)
    fixed_dofs_d = make_fixed_dofs_for_root_clamp(nodes_d, root_mask_d)
    f_d = distributed_vertical_load_to_nodes(nodes_d, span=wing.span, total_lift_newtons=total_lift, distribution="elliptic")

    try:
        res_d = solve_truss(
            nodes_d, edges_d, areas_d, E_modulus, f_d, fixed_dofs_d,
            regularization=1e-9
        )
        max_sigma_d = float(np.max(np.abs(res_d.member_stress)))
        tip_disp_d = float(res_d.tip_disp)
        lambda2_d = algebraic_connectivity_safe(len(nodes_d), edges_d)
        ebc_d = edge_betweenness_stats(len(nodes_d), edges_d)
        return {
            "success": True,
            "reason": "ok",
            "tip_disp": tip_disp_d,
            "max_sigma": max_sigma_d,
            "lambda2": lambda2_d,
            "ebc_max": ebc_d["max"],
            "nodes_d": nodes_d,
            "edges_d": edges_d,
        }
    except Exception as err:
        return {
            "success": False,
            "reason": f"solve_failed: {err}",
            "tip_disp": np.inf,
            "max_sigma": np.inf,
            "lambda2": 0.0,
            "ebc_max": np.inf,
            "nodes_d": nodes_d,
            "edges_d": edges_d,
        }


def plot_lattice(ax, nodes, edges, title, stride=1, color="C0"):
    ax.set_title(title)
    if len(edges) == 0:
        ax.text(0.5, 0.5, 0.5, "No edges", transform=ax.transAxes)
        return
    use = np.arange(0, len(edges), max(1, stride))
    for e in use:
        i, j = edges[e]
        p = nodes[i]
        q = nodes[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, linewidth=0.8)
    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z (depth)")


def main():
    # -----------------------------
    # Problem setup
    # -----------------------------
    wing = WingBox(span=4.0, chord=1.0, depth=0.16, root_tol=1e-9)

    E_modulus = 70e9       # Pa
    density = 2700.0       # kg/m^3
    sigma_allow = 250e6    # Pa
    u_tip_max = 0.25       # m, placeholder
    lambda2_min = 1e-3

    a_min = 1e-6           # m^2
    a_max = 5e-4           # m^2
    a_init = 5e-5          # m^2

    # lattice resolution
    nx, ny, nz = 7, 21, 5

    # load
    g = 9.80665
    m_vehicle = 20.0
    total_lift = 2.5 * m_vehicle * g

    # optimization loop settings
    n_iter = 20
    damage_mode = "khop"   # or "nearest_n"
    prune_after_iter = 3
    prune_area_threshold = 5.0e-6     # much less conservative than before
    prune_stress_fraction = 0.02

    # -----------------------------
    # Initial design
    # -----------------------------
    lat = generate_octet_ground_structure(
        span=wing.span,
        chord=wing.chord,
        depth=wing.depth,
        ny=ny, nx=nx, nz=nz
    )
    nodes = lat.nodes
    edges = lat.edges
    areas = np.full(len(edges), a_init, dtype=float)

    root_mask = wing.root_mask(nodes)
    root_nodes = np.where(root_mask)[0]
    fixed_dofs = make_fixed_dofs_for_root_clamp(nodes, root_mask)

    history = {
        "mass": [],
        "tip_disp": [],
        "max_sigma": [],
        "damaged_tip_disp": [],
        "damaged_max_sigma": [],
        "lambda2": [],
        "damaged_lambda2": [],
        "n_edges": [],
        "damage_success": [],
    }

    # -----------------------------
    # Optimization loop
    # -----------------------------
    for it in range(n_iter):
        f = distributed_vertical_load_to_nodes(
            nodes, span=wing.span,
            total_lift_newtons=total_lift,
            distribution="elliptic"
        )

        res = solve_truss(nodes, edges, areas, E_modulus, f, fixed_dofs, regularization=1e-9)
        max_sigma = float(np.max(np.abs(res.member_stress)))
        tip_disp = float(res.tip_disp)
        mass = estimate_mass(nodes, edges, areas, density)

        lambda2 = algebraic_connectivity_safe(len(nodes), edges)

        damage_info = evaluate_damage_case(
            wing=wing,
            nodes=nodes,
            edges=edges,
            areas=areas,
            E_modulus=E_modulus,
            total_lift=total_lift,
            damage_mode=damage_mode,
            khop_k=2,
            nearest_n=25,
        )

        history["mass"].append(mass)
        history["tip_disp"].append(tip_disp)
        history["max_sigma"].append(max_sigma)
        history["damaged_tip_disp"].append(damage_info["tip_disp"])
        history["damaged_max_sigma"].append(damage_info["max_sigma"])
        history["lambda2"].append(lambda2)
        history["damaged_lambda2"].append(damage_info["lambda2"])
        history["n_edges"].append(len(edges))
        history["damage_success"].append(damage_info["success"])

        intact_ok = (max_sigma <= sigma_allow) and (tip_disp <= u_tip_max)
        damaged_ok = (
            damage_info["success"] and
            damage_info["max_sigma"] <= sigma_allow and
            damage_info["tip_disp"] <= u_tip_max and
            damage_info["lambda2"] >= lambda2_min
        )

        print(
            f"Iter {it:02d} | mass={mass:.4f} kg | edges={len(edges):5d} | "
            f"tip={tip_disp:.4e} m | maxσ={max_sigma/1e6:.2f} MPa | "
            f"dam_tip={damage_info['tip_disp']:.4e} m | dam_maxσ={damage_info['max_sigma']/1e6:.2f} MPa | "
            f"damage_reason={damage_info['reason']} | ok(intact)={intact_ok} ok(dmg)={damaged_ok}"
        )

        # sizing update
        areas = simple_sizing_update(
            areas=areas,
            stress=res.member_stress,
            sigma_allow=sigma_allow,
            a_min=a_min,
            a_max=a_max,
            shrink_factor=0.985,
            grow_factor=1.06,
        )

        # pruning update
        if it >= prune_after_iter:
            edges, areas, keep_mask = prune_members(
                edges=edges,
                areas=areas,
                stress=res.member_stress,
                area_threshold=prune_area_threshold,
                stress_fraction_threshold=prune_stress_fraction,
                sigma_allow=sigma_allow,
                protect_root_nodes=root_nodes,
            )

    # -----------------------------
    # Final visualization
    # -----------------------------
    damage_info = evaluate_damage_case(
        wing=wing,
        nodes=nodes,
        edges=edges,
        areas=areas,
        E_modulus=E_modulus,
        total_lift=total_lift,
        damage_mode=damage_mode,
        khop_k=2,
        nearest_n=25,
    )

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    plot_lattice(ax1, nodes, edges, "Optimized intact lattice (subset of edges)", stride=max(1, len(edges) // 1000))

    ax2 = fig.add_subplot(122, projection="3d")
    plot_lattice(
        ax2,
        damage_info["nodes_d"],
        damage_info["edges_d"],
        f"Optimized damaged lattice (subset of edges)\nreason={damage_info['reason']}",
        stride=max(1, max(1, len(damage_info["edges_d"])) // 1000),
        color="C3",
    )

    plt.tight_layout()
    plt.show()

    # history plots
    fig2, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(history["mass"], marker="o")
    axs[0, 0].set_title("Mass vs iteration")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Mass (kg)")

    axs[0, 1].plot(np.array(history["max_sigma"]) / 1e6, marker="o", label="Intact")
    axs[0, 1].plot(np.array(history["damaged_max_sigma"]) / 1e6, marker="s", label="Damaged")
    axs[0, 1].axhline(sigma_allow / 1e6, color="r", linestyle="--", label="Allowable")
    axs[0, 1].set_title("Max stress vs iteration")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Stress (MPa)")
    axs[0, 1].legend()

    axs[1, 0].plot(history["tip_disp"], marker="o", label="Intact")
    axs[1, 0].plot(history["damaged_tip_disp"], marker="s", label="Damaged")
    axs[1, 0].axhline(u_tip_max, color="r", linestyle="--", label="Limit")
    axs[1, 0].set_title("Tip displacement vs iteration")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Tip displacement (m)")
    axs[1, 0].legend()

    axs[1, 1].plot(history["n_edges"], marker="o", label="# edges")
    axs[1, 1].plot(history["lambda2"], marker="o", label="λ2 intact")
    axs[1, 1].plot(history["damaged_lambda2"], marker="s", label="λ2 damaged")
    axs[1, 1].set_title("Topology / connectivity history")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
