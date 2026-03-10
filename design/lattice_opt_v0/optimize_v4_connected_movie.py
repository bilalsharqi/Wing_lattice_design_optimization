import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import imageio.v2 as imageio

from wingbox_domain import WingBox
from load_mapping import distributed_vertical_load_to_nodes
from truss_solver import solve_truss, choose_tip_node
from damage_models import filter_to_root_connected_intact, damage_and_check_full_connectivity
from gt_metrics import is_connected_safe, algebraic_connectivity_safe, edge_betweenness_stats

from generate_octet_lattice import generate_octet_ground_structure
from generate_square_lattice import generate_square_wingbox_lattice
from grouped_pruning import build_spanwise_bay_groups, evaluate_group_scores, prune_one_group


# ============================================================
# Utilities
# ============================================================

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


def build_lattice(wing, lattice_type):
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


# ============================================================
# Damage models for movie/debugging
# ============================================================

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
    """
    Mild damage mode for visualization/debugging.
    Removes edges attached to the seed node and, for k>0, edges attached to
    graph-neighbor nodes within k hops. Nodes are not explicitly deleted;
    after edge removal, the structure is filtered to the root-connected piece.
    """
    n_nodes = len(nodes)
    adjacency = _build_node_adjacency(n_nodes, edges)
    impacted_nodes = _bfs_nodes_within_k_hops(seed, adjacency, k)

    impacted_set = set(int(v) for v in impacted_nodes)
    remove_edge_mask = np.array(
        [(int(i) in impacted_set) or (int(j) in impacted_set) for i, j in edges],
        dtype=bool,
    )
    removed_edge_idx_original = np.where(remove_edge_mask)[0]

    kept_edges = edges[~remove_edge_mask]
    kept_areas = areas[~remove_edge_mask]

    return {
        "seed": int(seed),
        "removed_nodes_original_idx": impacted_nodes,
        "removed_edges_original_idx": removed_edge_idx_original,
        "edges_kept_pre_filter": kept_edges,
        "areas_kept_pre_filter": kept_areas,
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
    """
    Returns a dict with a damaged graph, plus bookkeeping in terms of the
    ORIGINAL current-iteration geometry.

    Modes:
      - local_edge_ring: mild local edge removal around seed using graph hops
      - legacy_graph_hops: original node-removal damage operator
    """
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
            [
                (int(i) in removed_node_set) or (int(j) in removed_node_set)
                for i, j in edges
            ],
            dtype=bool,
        )
        removed_edges_original_idx = np.where(removed_edge_mask)[0]

        info["removed_edges_original_idx"] = removed_edges_original_idx
        info["damage_mode"] = damage_mode
        info["seed"] = int(seed)
        return info

    if damage_mode == "local_edge_ring":
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

    raise ValueError(f"Unknown damage_mode='{damage_mode}'")


# ============================================================
# Grouped pruning
# ============================================================

def try_grouped_prune_and_validate(
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
    """
    Evaluate candidate grouped pruning on the current intact design.
    Returns the *next iteration* intact design if an acceptable prune exists.
    The returned state should not be used to overwrite the current saved frame;
    save current evaluated state first, then advance to returned state.
    """
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
            nodes,
            trial_edges,
            trial_areas,
            wing,
        )

        if not is_connected_safe(len(trial_nodes), trial_edges):
            print(f"    reject {glabel}: intact disconnected")
            continue

        trial_root_mask = wing.root_mask(trial_nodes)
        trial_fixed_dofs = make_fixed_dofs_for_root_clamp(trial_nodes, trial_root_mask)

        try:
            f_trial = distributed_vertical_load_to_nodes(
                trial_nodes,
                wing.span,
                total_lift,
                distribution="elliptic",
            )
            trial_tip = choose_tip_node(trial_nodes, wing)
            trial_res = solve_truss(
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
            f_trial_d = distributed_vertical_load_to_nodes(
                trial_nodes_d,
                wing.span,
                total_lift,
                distribution="elliptic",
            )
            trial_tip_d = choose_tip_node(trial_nodes_d, wing)
            trial_res_d = solve_truss(
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


# ============================================================
# Plot helpers
# ============================================================

def _set_equal_3d_axes(ax, xyz_min, xyz_max):
    xr = xyz_max[0] - xyz_min[0]
    yr = xyz_max[1] - xyz_min[1]
    zr = xyz_max[2] - xyz_min[2]
    r = 0.5 * max(xr, yr, zr, 1e-9)
    cx = 0.5 * (xyz_min[0] + xyz_max[0])
    cy = 0.5 * (xyz_min[1] + xyz_max[1])
    cz = 0.5 * (xyz_min[2] + xyz_max[2])
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
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=cmap(norm(vals[k])),
            linewidth=lw,
            alpha=alpha,
        )


def plot_removed_edge_coords(ax, edge_coords, color="k", lw=2.8, alpha=1.0, ls="-"):
    if edge_coords is None or len(edge_coords) == 0:
        return
    for seg in edge_coords:
        p, q = seg
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
        )


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
    if damaged_state["nodes"].size > 0:
        all_pts.append(damaged_state["nodes"])
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

    fig.suptitle("Pruning + damage evolution", fontsize=22, y=0.98)

    # Intact
    plot_edges_by_stress(
        ax_i,
        intact_state["nodes"],
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
    plot_single_node(ax_i, intact_state["tip_coord"], color="#2ca02c", marker="o", size=110, label="tip node")
    plot_single_node(ax_i, intact_state["max_span_coord"], color="#d62728", marker="^", size=130, label="max span")
    ax_i.set_title(f"Intact | iter {it:02d}", fontsize=18)
    ax_i.set_xlabel("x")
    ax_i.set_ylabel("y")
    ax_i.set_zlabel("z")
    _set_equal_3d_axes(ax_i, xyz_min, xyz_max)
    ax_i.view_init(elev=18, azim=-62)
    ax_i.legend(loc="upper left", fontsize=12)

    # Damaged
    if damaged_state["nodes"].size > 0 and damaged_state["edges"].size > 0:
        plot_edges_by_stress(
            ax_d,
            damaged_state["nodes"],
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
        plot_single_node(ax_d, damaged_state["tip_coord"], color="#2ca02c", marker="o", size=110, label="tip node")
        plot_single_node(ax_d, damaged_state["max_span_coord"], color="#d62728", marker="^", size=130, label="max span")
    scatter_nodes(ax_d, removed_damage_node_coords, color="#9467bd", marker="x", size=100, label="removed damage nodes")
    ax_d.set_title(f"Damaged | iter {it:02d}", fontsize=18)
    ax_d.set_xlabel("x")
    ax_d.set_ylabel("y")
    ax_d.set_zlabel("z")
    _set_equal_3d_axes(ax_d, xyz_min, xyz_max)
    ax_d.view_init(elev=18, azim=-62)
    ax_d.legend(loc="upper left", fontsize=12)

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.20, 0.015, 0.62])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$|\sigma| / \sigma_{allow}$", fontsize=13)

    ax_txt.text(
        0.01,
        0.98,
        info_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=14,
    )

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def assemble_gif_and_mp4(frame_dir, gif_path, mp4_path, fps=2):
    import numpy as np
    import imageio.v2 as imageio

    frame_files = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.lower().endswith(".png")]
    )
    if not frame_files:
        return False, False

    images = [imageio.imread(f) for f in frame_files]

    # Save GIF directly
    imageio.mimsave(gif_path, images, duration=1.0 / max(float(fps), 1.0))

    # Pad frames to even width/height for libx264 / yuv420p
    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        pad_h = h % 2
        pad_w = w % 2

        if pad_h == 0 and pad_w == 0:
            padded_images.append(img)
            continue

        if img.ndim == 3:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        else:
            pad_width = ((0, pad_h), (0, pad_w))

        img_pad = np.pad(img, pad_width, mode="edge")
        padded_images.append(img_pad)

    mp4_ok = True
    try:
        with imageio.get_writer(
            mp4_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_log_level="warning",
        ) as writer:
            for img in padded_images:
                writer.append_data(img)
    except Exception as err:
        mp4_ok = False
        print(f"MP4 save failed: {err}")

    return True, mp4_ok


def save_history_plots(hist, sigma_allow, u_tip_max, lattice_type, base_dir):
    if len(hist["mass"]) == 0:
        return None, None

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    axs[0, 0].plot(hist["mass"], marker="o", label="Mass")
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

    summary_svg = os.path.join(base_dir, f"lattice_opt_summary_{lattice_type}.svg")
    summary_png = os.path.join(base_dir, f"lattice_opt_summary_{lattice_type}.png")

    fig.savefig(summary_png, dpi=200, bbox_inches="tight")
    fig.savefig(summary_svg, bbox_inches="tight")
    plt.close(fig)

    return summary_svg, summary_png


# ============================================================
# Main
# ============================================================

def main():
    wing = WingBox(span=4.0, chord=1.0, depth=0.16, root_tol=1e-9)

    lattice_type = "square"
    # lattice_type = "octet"

    # Material / design
    E_modulus = 70e9
    density = 2700.0
    sigma_allow = 250e6
    u_tip_max = 0.25
    cond_max = 1e7

    a_min = 5e-7
    a_max = 5e-4
    a_init = 3.175e-5 # 1/8th inches

    # Load
    g = 9.80665
    m_vehicle = 50.0
    load_factor = 5.0
    total_lift = load_factor * m_vehicle * g

    # Loop settings
    n_iter = 25
    prune_after_iter = 1

    # Damage settings
    damage_mode = "local_edge_ring"   # "local_edge_ring" or "legacy_graph_hops"
    damage_k = 0                      # for local_edge_ring: 0 = seed-only edge ring, 1 = one graph ring, ...
    seed_span_fraction = 0.60
    seed_x_fraction = 0.50

    # Outputs
    base_dir = os.getcwd()
    h5_path = os.path.join(base_dir, f"lattice_opt_debug_v4_{lattice_type}.h5")
    frame_dir = os.path.join(base_dir, f"lattice_opt_frames_{lattice_type}")
    gif_path = os.path.join(base_dir, f"lattice_opt_evolution_{lattice_type}.gif")
    mp4_path = os.path.join(base_dir, f"lattice_opt_evolution_{lattice_type}.mp4")

    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # Build initial structure
    nodes, edges = build_lattice(wing, lattice_type)
    areas = np.full(len(edges), a_init, dtype=float)
    nodes, edges, areas, _, _ = filter_to_root_connected_intact(nodes, edges, areas, wing)

    print("Initial design summary:")
    print(f"  lattice type   = {lattice_type}")
    print(f"  nodes          = {len(nodes)}")
    print(f"  edges          = {len(edges)}")
    print(f"  total lift     = {total_lift:.2f} N")
    print(f"  damage mode    = {damage_mode}")
    print(f"  damage k       = {damage_k}")
    print(f"  HDF5 output    = {os.path.abspath(h5_path)}")
    print(f"  frame dir      = {os.path.abspath(frame_dir)}")

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
        meta.attrs["damage_mode"] = damage_mode
        meta.attrs["damage_k"] = damage_k
        meta.attrs["seed_span_fraction"] = seed_span_fraction
        meta.attrs["seed_x_fraction"] = seed_x_fraction

        for it in range(n_iter):
            intact_connected = is_connected_safe(len(nodes), edges)
            if not intact_connected:
                print(f"Iter {it:02d} | intact graph disconnected -> FAIL")
                break

            # -----------------------------
            # Evaluate CURRENT intact state
            # -----------------------------
            root_mask = wing.root_mask(nodes)
            fixed_dofs = make_fixed_dofs_for_root_clamp(nodes, root_mask)
            f = distributed_vertical_load_to_nodes(nodes, wing.span, total_lift, distribution="elliptic")
            tip_node_idx = choose_tip_node(nodes, wing)
            tip_coord = nodes[tip_node_idx]
            max_span_node_idx = int(np.argmax(nodes[:, 1]))
            max_span_coord = nodes[max_span_node_idx]

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

            # -----------------------------
            # Evaluate CURRENT damaged state
            # -----------------------------
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

            if (not damaged_connected) or (not damaged_screen_passed):
                damage_reason = "damaged_graph_disconnected_or_screen_failed"
            else:
                nodes_d = damage_info["nodes_d"]
                edges_d = damage_info["edges_d"]
                areas_d = damage_info["areas_d"]
                root_mask_d = wing.root_mask(nodes_d)
                fixed_dofs_d = make_fixed_dofs_for_root_clamp(nodes_d, root_mask_d)
                f_d = distributed_vertical_load_to_nodes(nodes_d, wing.span, total_lift, distribution="elliptic")
                dam_tip_node_idx = choose_tip_node(nodes_d, wing)
                dam_tip_coord = nodes_d[dam_tip_node_idx]
                dam_max_span_node_idx = int(np.argmax(nodes_d[:, 1]))
                dam_max_span_coord = nodes_d[dam_max_span_node_idx]

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
                    dam_lambda2 = algebraic_connectivity_safe(len(nodes_d), edges_d)
                    dam_ebc = edge_betweenness_stats(len(nodes_d), edges_d)

                    if (not np.isfinite(dam_cond)) or (dam_cond > cond_max):
                        damage_reason = "damaged_solver_instability"
                    else:
                        dam_tip = float(res_d.tip_disp)
                        dam_sigma = float(np.max(np.abs(res_d.member_stress)))
                except Exception as err:
                    damage_reason = f"damaged_solve_failed:{err}"

            removed_damage_node_idx = np.asarray(
                damage_info.get("removed_nodes_original_idx", np.zeros((0,), dtype=int)),
                dtype=int,
            )
            removed_damage_node_coords = (
                nodes[removed_damage_node_idx]
                if removed_damage_node_idx.size > 0
                else np.zeros((0, 3), dtype=float)
            )
            removed_damage_edge_idx = np.asarray(
                damage_info.get("removed_edges_original_idx", np.zeros((0,), dtype=int)),
                dtype=int,
            )
            removed_damage_edge_coords = _edge_coords_from_original(nodes, edges, removed_damage_edge_idx)

            # -----------------------------
            # Save coherent CURRENT iteration state
            # -----------------------------
            intact_util = max_sigma / sigma_allow
            damaged_util = dam_sigma / sigma_allow if np.isfinite(dam_sigma) else np.inf

            info_text = (
                f"iter={it:02d}\n"
                f"edges={len(edges):3d}  pruned_prev={len(prev_prune_removed_edge_coords):1d}\n"
                f"mass={mass:7.3f} kg\n"
                f"sigma_i={max_sigma/1e6:7.3f} MPa   sigma_d={dam_sigma/1e6 if np.isfinite(dam_sigma) else np.inf:7.3f} MPa\n"
                f"u_tip_i={tip_disp:8.4e} m   u_tip_d={dam_tip:8.4e} m\n"
                f"tip_y_i={tip_coord[1]:5.3f}   ymax_i={max_span_coord[1]:5.3f}\n"
                f"tip_y_d={dam_tip_coord[1] if np.isfinite(dam_tip_coord[1]) else np.nan:5.3f}   ymax_d={dam_max_span_coord[1] if np.isfinite(dam_max_span_coord[1]) else np.nan:5.3f}\n"
                f"damage_mode={damage_mode}  damage_k={damage_k}\n"
                f"damage_reason={damage_reason}\n"
                f"prev_prune={prev_prune_label}"
            )

            frame_png = os.path.join(frame_dir, f"frame_{it:04d}.png")
            frame_svg = os.path.join(frame_dir, f"frame_{it:04d}.svg")

            intact_state = {
                "nodes": nodes,
                "edges": edges,
                "member_stress": res.member_stress,
                "tip_coord": tip_coord,
                "max_span_coord": max_span_coord,
            }
            damaged_state = {
                "nodes": nodes_d,
                "edges": edges_d,
                "member_stress": (
                    res_d.member_stress
                    if res_d is not None and np.isfinite(dam_sigma)
                    else np.zeros((len(edges_d),), dtype=float)
                ),
                "tip_coord": dam_tip_coord,
                "max_span_coord": dam_max_span_coord,
            }

            make_movie_frame(
                out_png=frame_png,
                out_svg=frame_svg,
                wing=wing,
                it=it,
                sigma_allow=sigma_allow,
                intact_state=intact_state,
                damaged_state=damaged_state,
                root_mask_intact=root_mask,
                root_mask_damaged=root_mask_d,
                removed_edges_prune_prev_coords=prev_prune_removed_edge_coords,
                removed_edges_damage_coords=removed_damage_edge_coords,
                removed_damage_node_coords=removed_damage_node_coords,
                info_text=info_text,
            )

            save_iteration_hdf5(
                h5,
                it,
                {
                    # intact evaluated state
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
                    "tip_minus_max_span_y": float(tip_coord[1] - max_span_coord[1]),

                    # damaged evaluated state
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
                    "damaged_tip_y": float(dam_tip_coord[1]) if np.isfinite(dam_tip_coord[1]) else np.nan,
                    "damaged_max_span_node_idx": int(dam_max_span_node_idx),
                    "damaged_max_span_coord": dam_max_span_coord,
                    "damaged_max_span_y": float(dam_max_span_coord[1]) if np.isfinite(dam_max_span_coord[1]) else np.nan,
                    "damaged_tip_minus_max_span_y": (
                        float(dam_tip_coord[1] - dam_max_span_coord[1])
                        if np.isfinite(dam_tip_coord[1]) and np.isfinite(dam_max_span_coord[1])
                        else np.nan
                    ),
                    "removed_nodes_damage_original_idx": removed_damage_node_idx,
                    "removed_nodes_damage_original_coords": removed_damage_node_coords,
                    "removed_edges_damage_original_idx": removed_damage_edge_idx,
                    "removed_edges_damage_original_coords": removed_damage_edge_coords,

                    # previous accepted prune, shown on THIS frame in black
                    "removed_edges_prune_prev_coords": prev_prune_removed_edge_coords,
                    "prev_prune_label": prev_prune_label,

                    # frame paths
                    "frame_png": frame_png,
                    "frame_svg": frame_svg,
                },
            )

            print(
                f"Iter {it:02d} | edges={len(edges):5d} | mass={mass:8.2f} kg | "
                f"σ_intact={max_sigma/1e6:7.2f} MPa ({intact_util:5.3f}) | "
                f"σ_dmg={dam_sigma/1e6 if np.isfinite(dam_sigma) else np.inf:7.2f} MPa ({damaged_util:5.3f}) | "
                f"u_tip={tip_disp:8.4e} m | u_tip_d={dam_tip:8.4e} m | "
                f"tip_y_i={tip_coord[1]:6.3f} ymax_i={max_span_coord[1]:6.3f} | "
                f"tip_y_d={dam_tip_coord[1] if np.isfinite(dam_tip_coord[1]) else np.nan:6.3f} ymax_d={dam_max_span_coord[1] if np.isfinite(dam_max_span_coord[1]) else np.nan:6.3f} | "
                f"conn_i={intact_connected} conn_d={damaged_connected} | "
                f"cond_i={res.cond_est:.3e} cond_d={dam_cond:.3e} | "
                f"{damage_reason} | prev_prune={prev_prune_label}"
            )

            hist["mass"].append(mass)
            hist["tip_intact"].append(tip_disp)
            hist["tip_damaged"].append(dam_tip)
            hist["sigma_intact"].append(max_sigma / 1e6)
            hist["sigma_damaged"].append(dam_sigma / 1e6 if np.isfinite(dam_sigma) else np.nan)
            hist["n_edges"].append(len(edges))
            hist["pruned_count_prev"].append(len(prev_prune_removed_edge_coords))

            # -----------------------------
            # Advance to NEXT iteration state
            # -----------------------------
            areas_after_sizing = simple_sizing_update(
                areas, res.member_stress, sigma_allow, a_min, a_max
            )

            if it >= prune_after_iter:
                prune_out = try_grouped_prune_and_validate(
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

    # Assemble movie from PNG frames
    gif_ok, mp4_ok = assemble_gif_and_mp4(frame_dir, gif_path, mp4_path, fps=2)

    # Save final history plot as SVG + PNG
    summary_svg, summary_png = save_history_plots(
        hist=hist,
        sigma_allow=sigma_allow,
        u_tip_max=u_tip_max,
        lattice_type=lattice_type,
        base_dir=base_dir,
    )

    print(f"\nSaved debug data to HDF5: {os.path.abspath(h5_path)}")
    if gif_ok:
        print(f"Saved GIF: {os.path.abspath(gif_path)}")
    if mp4_ok:
        print(f"Saved MP4: {os.path.abspath(mp4_path)}")
    elif gif_ok:
        print("MP4 writer not available; GIF was still saved.")

    if summary_svg is not None:
        print(f"Saved summary SVG: {os.path.abspath(summary_svg)}")
    if summary_png is not None:
        print(f"Saved summary PNG: {os.path.abspath(summary_png)}")


if __name__ == "__main__":
    main()