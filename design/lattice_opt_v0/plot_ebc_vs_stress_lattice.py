import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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


STRESS_DATASET = "member_stress"

EDGE_WEIGHT_MODES = ["length", "area_scaled", "compliance"]
DEFAULT_E = 70e9

ROOT_AXIS = 1
ROOT_TOL = None

# Plot view axes: 0=x, 1=y, 2=z
AX0 = 0
AX1 = 1
LINEWIDTH = 1.5


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


def make_lc(ax, segs, values, cmap, linewidth):
    lc = LineCollection(segs, cmap=cmap, linewidths=linewidth)
    lc.set_array(values)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    return lc


def plot_ebc_vs_stress_lattice(H5_FILE, ITERATION="final", OUTPUT_DIR=""):
    if not H5_FILE:
        raise RuntimeError("H5_FILE must be provided")

    out_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.join(
        os.path.dirname(os.path.abspath(H5_FILE)),
        "gt_metric_visual_all_weights"
    )
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(H5_FILE, "r") as h5:
        key = resolve_iter_key(h5, ITERATION)
        grp = h5[key]

        nodes = grp["nodes"][()]
        edges = grp["edges"][()].astype(int)
        stress = np.asarray(grp[STRESS_DATASET][()]).reshape(-1)
        force_vector = np.asarray(grp["force_vector"][()]).reshape(-1)

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

    source_nodes, source_weights = loaded_source_nodes(
        force_vector,
        n_nodes=nodes.shape[0],
        dof_per_node=None,
        component=2,
        atol=1e-12,
    )
    target_nodes = root_boundary_nodes(nodes, axis=ROOT_AXIS, tol=ROOT_TOL)

    abs_stress = np.abs(stress)
    stress_norm = abs_stress / abs_stress.max() if abs_stress.max() > 0.0 else abs_stress.copy()

    segs = np.asarray([
        [[nodes[i, AX0], nodes[i, AX1]], [nodes[j, AX0], nodes[j, AX1]]]
        for i, j in edges
    ], dtype=float)

    ebc_values = []
    for mode in EDGE_WEIGHT_MODES:
        vals = edge_boundary_betweenness_load_weighted(
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
        ebc_values.append(vals)

    fig, axs = plt.subplots(1, 4, figsize=(23, 6), constrained_layout=True)

    for ax, mode, vals in zip(axs[:3], EDGE_WEIGHT_MODES, ebc_values):
        lc = make_lc(ax, segs, vals, "viridis", LINEWIDTH)
        ax.set_title(rf"$EBC_{{LB}}$ ({mode})")
        ax.set_xlabel(f"axis {AX0}")
        ax.set_ylabel(f"axis {AX1}")
        fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)

    lc = make_lc(axs[3], segs, stress_norm, "plasma", LINEWIDTH)
    axs[3].set_title(r"$|\sigma_e|$ (normalized)")
    axs[3].set_xlabel(f"axis {AX0}")
    axs[3].set_ylabel(f"axis {AX1}")
    fig.colorbar(lc, ax=axs[3], fraction=0.046, pad=0.04)

    png = os.path.join(out_dir, f"ebc_vs_stress_lattice_all_weights_{key}.png")
    fig.savefig(png, dpi=220)
    plt.close(fig)

    print("Saved:")
    print(png)


if __name__ == "__main__":
    main()