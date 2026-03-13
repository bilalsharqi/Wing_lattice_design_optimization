import os
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from gt_metrics_updated import (
    edge_boundary_betweenness_load_weighted,
    loaded_source_nodes,
    root_boundary_nodes,
)

# ============================================================
# USER SETTINGS
# ============================================================

# H5_FILE = r"results\lattice_opt_voronoi_20260312_142430\beam\data\lattice_opt_beam_voronoi.h5"
# ITERATION = "final"          # "initial", "final", or integer like 12
# OUTPUT_DIR = r""       # optional; if blank, creates a folder next to H5

# H5_FILE = r"results\lattice_opt_two_skin_graded_hex_20260311_193633\beam\data\lattice_opt_beam_two_skin_graded_hex.h5"
# ITERATION = "final"          # "initial", "final", or integer like 12
# OUTPUT_DIR = r""       # optional; if blank, creates a folder next to H5

H5_FILE = r"results\lattice_opt_graded_hex_20260311_195142\beam\data\lattice_opt_beam_graded_hex.h5"
ITERATION = "final"          # "initial", "final", or integer like 12
OUTPUT_DIR = r""       # optional; if blank, creates a folder next to H5

# Which stress field to use from the H5:
STRESS_DATASET = "member_stress"

# Weighting modes to evaluate
EDGE_WEIGHT_MODES = ["length", "area_scaled", "compliance"]
DEFAULT_E = 70e9

# Force/source interpretation
FORCE_COMPONENT = 2          # use Fz as source weighting
SOURCE_ATOL = 1e-12

# Root boundary detection
ROOT_AXIS = 1                # spanwise y-axis
ROOT_TOL = None              # auto if None

# Top-k overlap fraction
TOP_FRAC = 0.10

# Plot settings
PLOT_LOGX = False
PLOT_LOGY = False
POINT_SIZE = 18


# ============================================================
# Helpers
# ============================================================

def resolve_iter_key(h5, iteration):
    keys = sorted([k for k in h5.keys() if k.startswith("iter_")])
    if not keys:
        raise ValueError("No iteration groups found in HDF5")
    if isinstance(iteration, str):
        val = iteration.strip().lower()
        if val == "initial":
            return keys[0]
        if val == "final":
            return keys[-1]
        if val.isdigit():
            key = f"iter_{int(val):04d}"
            if key not in h5:
                raise ValueError(f"Iteration {key} not found")
            return key
        raise ValueError("ITERATION must be 'initial', 'final', or integer")
    key = f"iter_{int(iteration):04d}"
    if key not in h5:
        raise ValueError(f"Iteration {key} not found")
    return key


def rank_array_desc(x):
    order = np.argsort(-x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))
    return ranks


def top_overlap_fraction(a, b, frac=0.10):
    n = len(a)
    k = max(1, int(math.ceil(float(frac) * n)))
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return len(ia.intersection(ib)) / float(k), k


def analyze_one_mode(out_dir, mode, nodes, edges, stress, force_vector, areas, E, source_nodes, source_weights, target_nodes):
    ebc_lb = edge_boundary_betweenness_load_weighted(
        nodes=nodes,
        edges=edges,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_weights=source_weights,
        edge_weight_mode=mode,
        areas=areas,
        E=E,
        normalized=True,
    )

    abs_stress = np.abs(stress)
    if len(abs_stress) != len(ebc_lb):
        raise ValueError("Stress array length does not match number of edges")

    rho, pval = spearmanr(abs_stress, ebc_lb)
    overlap, k = top_overlap_fraction(abs_stress, ebc_lb, frac=TOP_FRAC)

    summary_txt = os.path.join(out_dir, f"ebc_lb_summary_{mode}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"H5_FILE: {H5_FILE}\n")
        f.write(f"ITERATION: {ITERATION}\n")
        f.write(f"edge_weight_mode: {mode}\n")
        f.write(f"n_nodes: {nodes.shape[0]}\n")
        f.write(f"n_edges: {edges.shape[0]}\n")
        f.write(f"n_sources: {len(source_nodes)}\n")
        f.write(f"n_targets: {len(target_nodes)}\n")
        f.write(f"spearman_rho: {float(rho):.6f}\n")
        f.write(f"spearman_pvalue: {float(pval):.6e}\n")
        f.write(f"top_frac: {float(TOP_FRAC):.4f}\n")
        f.write(f"top_k: {int(k)}\n")
        f.write(f"top_overlap_fraction: {float(overlap):.6f}\n")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(ebc_lb, abs_stress, s=POINT_SIZE, alpha=0.75)
    ax.set_xlabel(f"Load-boundary edge betweenness, $EBC_{{LB}}$ ({mode})")
    ax.set_ylabel(r"$|\sigma_e|$")
    ax.set_title(f"Stress vs $EBC_{{LB}}$ ({mode})")
    if PLOT_LOGX:
        ax.set_xscale("log")
    if PLOT_LOGY:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.97,
        f"Spearman $\\rho$ = {float(rho):.3f}\nTop {TOP_FRAC:.0%} overlap = {float(overlap):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
    )
    scatter_png = os.path.join(out_dir, f"stress_vs_ebc_lb_{mode}.png")
    fig.tight_layout()
    fig.savefig(scatter_png, dpi=200)
    plt.close(fig)

    r_stress = rank_array_desc(abs_stress)
    r_ebc = rank_array_desc(ebc_lb)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(r_ebc, r_stress, s=POINT_SIZE, alpha=0.75)
    ax.set_xlabel(rf"Rank of $EBC_{{LB}}$ ({mode}) (0 = highest)")
    ax.set_ylabel(r"Rank of $|\sigma_e|$ (0 = highest)")
    ax.set_title(f"Rank correlation ({mode})")
    ax.grid(True, alpha=0.3)
    rank_png = os.path.join(out_dir, f"rank_stress_vs_ebc_lb_{mode}.png")
    fig.tight_layout()
    fig.savefig(rank_png, dpi=200)
    plt.close(fig)

    np.savez(
        os.path.join(out_dir, f"ebc_lb_data_{mode}.npz"),
        nodes=nodes,
        edges=edges,
        stress=stress,
        abs_stress=abs_stress,
        force_vector=force_vector,
        source_nodes=source_nodes,
        source_weights=source_weights,
        target_nodes=target_nodes,
        ebc_lb=ebc_lb,
        stress_rank=r_stress,
        ebc_rank=r_ebc,
    )

    return summary_txt, scatter_png, rank_png, os.path.join(out_dir, f"ebc_lb_data_{mode}.npz")


# ============================================================
# Main
# ============================================================

def main():
    if not H5_FILE:
        raise RuntimeError("Set H5_FILE at the top of the script")
    out_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.join(os.path.dirname(os.path.abspath(H5_FILE)), "gt_metric_analysis")
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(H5_FILE, "r") as h5:
        key = resolve_iter_key(h5, ITERATION)
        grp = h5[key]

        areas = None
        if "areas" in grp.keys():
            areas = np.asarray(grp["areas"][:]).reshape(-1)
        elif "areas" in h5.keys():
            areas = np.asarray(h5["areas"][:]).reshape(-1)

        E = DEFAULT_E
        if "E" in grp.keys():
            E = float(np.asarray(grp["E"][()]).reshape(-1)[0])
        elif "E" in h5.keys():
            E = float(np.asarray(h5["E"][()]).reshape(-1)[0])

        nodes = grp["nodes"][()]
        edges = grp["edges"][()].astype(int)
        stress = np.asarray(grp[STRESS_DATASET][()]).reshape(-1)
        force_vector = np.asarray(grp["force_vector"][()]).reshape(-1)

    source_nodes, source_weights = loaded_source_nodes(
        force_vector,
        n_nodes=nodes.shape[0],
        dof_per_node=None,
        component=FORCE_COMPONENT,
        atol=SOURCE_ATOL,
    )
    target_nodes = root_boundary_nodes(nodes, axis=ROOT_AXIS, tol=ROOT_TOL)

    generated = []
    for mode in EDGE_WEIGHT_MODES:
        generated.extend(
            analyze_one_mode(
                out_dir, mode, nodes, edges, stress, force_vector, areas, E,
                source_nodes, source_weights, target_nodes
            )
        )

    print("Saved:")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()