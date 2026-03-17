import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from gt_metrics_updated import (
    edge_boundary_betweenness_load_weighted,
    loaded_source_nodes,
    root_boundary_nodes,
)

# ============================================================
# USER SETTINGS
# ============================================================

H5_FILE = r"results\Voronoi\voronoi_area_ON_EBC_veto_ON_prune_mode_random\beam\data\lattice_opt_beam_voronoi.h5"
OUTPUT_DIR = r""          # if blank, makes folder next to H5
FPS = 2

# Which EBC weighting to visualize over time:
# "length", "area_scaled", "compliance"
EBC_WEIGHT_MODE = "length"

# 2D plot view axes (top view by default: x-y plane)
AX0 = 0   # x
AX1 = 1   # y

LINEWIDTH = 1.6
ROOT_AXIS = 1
ROOT_TOL = None
DEFAULT_E = 70e9

# Color normalization mode:
# "global" -> one color scale across all iterations
# "per_iter" -> each frame normalized independently
COLOR_NORM_MODE = "global"

# Also overlay normalized stress side-by-side
INCLUDE_STRESS_PANEL = True

# Frame filename prefix
FRAME_PREFIX = "ebc_frame"


def resolve_iter_keys(h5):
    keys = sorted([k for k in h5.keys() if k.startswith("iter_")])
    if not keys:
        raise ValueError("No iteration groups found in HDF5")
    return keys


def make_output_dir(h5_file, output_dir):
    if output_dir:
        out_dir = output_dir
    else:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(h5_file)),
            f"gt_metric_evolution_{EBC_WEIGHT_MODE}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_E_value(h5, grp, default_E=70e9):
    if "E" in grp.keys():
        return float(np.asarray(grp["E"][()]).reshape(-1)[0])
    if "E" in h5.keys():
        return float(np.asarray(h5["E"][()]).reshape(-1)[0])
    return float(default_E)


def get_areas(h5, grp):
    if "areas" in grp.keys():
        return np.asarray(grp["areas"][:]).reshape(-1)
    if "areas" in h5.keys():
        return np.asarray(h5["areas"][:]).reshape(-1)
    return None


def build_segments(nodes, edges, ax0=0, ax1=1):
    return np.asarray([
        [[nodes[i, ax0], nodes[i, ax1]], [nodes[j, ax0], nodes[j, ax1]]]
        for i, j in edges
    ], dtype=float)


def make_line_collection(ax, segs, values, cmap="viridis", linewidth=1.5, norm=None):
    lc = LineCollection(segs, cmap=cmap, linewidths=linewidth, norm=norm)
    lc.set_array(values)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    return lc


def compute_edge_ebc(nodes, edges, areas, force_vector, E, edge_weight_mode):
    source_nodes, source_weights = loaded_source_nodes(
        force_vector,
        n_nodes=nodes.shape[0],
        dof_per_node=None,
        component=2,
        atol=1e-12,
    )
    target_nodes = root_boundary_nodes(nodes, axis=ROOT_AXIS, tol=ROOT_TOL)

    ebc = edge_boundary_betweenness_load_weighted(
        nodes=nodes,
        edges=edges,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_weights=source_weights,
        edge_weight_mode=edge_weight_mode,
        areas=areas,
        E=E,
        normalized=True,
    )
    return np.asarray(ebc).reshape(-1)


def normalize_stress(stress):
    abs_stress = np.abs(np.asarray(stress).reshape(-1))
    smax = float(np.max(abs_stress)) if abs_stress.size > 0 else 0.0
    return abs_stress / smax if smax > 0.0 else abs_stress.copy()


def main():
    if not H5_FILE:
        raise RuntimeError("Set H5_FILE at the top of the script")

    out_dir = make_output_dir(H5_FILE, OUTPUT_DIR)

    with h5py.File(H5_FILE, "r") as h5:
        iter_keys = resolve_iter_keys(h5)

        frame_data = []
        global_ebc_max = 0.0
        global_stress_max = 0.0

        for key in iter_keys:
            grp = h5[key]
            nodes = np.asarray(grp["nodes"][()])
            edges = np.asarray(grp["edges"][()]).astype(int)
            stress = np.asarray(grp["member_stress"][()]).reshape(-1)
            force_vector = np.asarray(grp["force_vector"][()]).reshape(-1)

            areas = get_areas(h5, grp)
            E = get_E_value(h5, grp, default_E=DEFAULT_E)

            ebc = compute_edge_ebc(
                nodes=nodes,
                edges=edges,
                areas=areas,
                force_vector=force_vector,
                E=E,
                edge_weight_mode=EBC_WEIGHT_MODE,
            )
            stress_norm = normalize_stress(stress)
            segs = build_segments(nodes, edges, ax0=AX0, ax1=AX1)

            frame_data.append({
                "key": key,
                "segs": segs,
                "ebc": ebc,
                "stress_norm": stress_norm,
            })

            if ebc.size > 0:
                global_ebc_max = max(global_ebc_max, float(np.max(ebc)))
            if stress_norm.size > 0:
                global_stress_max = max(global_stress_max, float(np.max(stress_norm)))

    ebc_norm_global = Normalize(vmin=0.0, vmax=max(global_ebc_max, 1e-12))
    stress_norm_global = Normalize(vmin=0.0, vmax=max(global_stress_max, 1e-12))

    frame_paths = []

    for idx, item in enumerate(frame_data):
        key = item["key"]
        segs = item["segs"]
        ebc = item["ebc"]
        stress_norm = item["stress_norm"]

        if INCLUDE_STRESS_PANEL:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            ax_ebc, ax_stress = axs
        else:
            fig, ax_ebc = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
            ax_stress = None

        if COLOR_NORM_MODE == "global":
            ebc_norm = ebc_norm_global
            s_norm = stress_norm_global
        else:
            ebc_norm = Normalize(vmin=0.0, vmax=max(float(np.max(ebc)) if ebc.size else 0.0, 1e-12))
            s_norm = Normalize(vmin=0.0, vmax=max(float(np.max(stress_norm)) if stress_norm.size else 0.0, 1e-12))

        lc0 = make_line_collection(ax_ebc, segs, ebc, cmap="viridis", linewidth=LINEWIDTH, norm=ebc_norm)
        ax_ebc.set_title(rf"$EBC_{{LB}}$ ({EBC_WEIGHT_MODE}) | {key}")
        ax_ebc.set_xlabel("x")
        ax_ebc.set_ylabel("y")
        cb0 = fig.colorbar(lc0, ax=ax_ebc, fraction=0.046, pad=0.04)
        cb0.set_label(r"$EBC_{LB}$")

        if INCLUDE_STRESS_PANEL:
            lc1 = make_line_collection(ax_stress, segs, stress_norm, cmap="plasma", linewidth=LINEWIDTH, norm=s_norm)
            ax_stress.set_title(rf"Normalized $|\sigma|$ | {key}")
            ax_stress.set_xlabel("x")
            ax_stress.set_ylabel("y")
            cb1 = fig.colorbar(lc1, ax=ax_stress, fraction=0.046, pad=0.04)
            cb1.set_label(r"$|\sigma| / |\sigma|_{\max}$")

        png = os.path.join(out_dir, f"{FRAME_PREFIX}_{idx:04d}_{key}.png")
        fig.savefig(png, dpi=220)
        plt.close(fig)
        frame_paths.append(png)

    gif_path = os.path.join(out_dir, f"ebc_evolution_{EBC_WEIGHT_MODE}.gif")
    mp4_path = os.path.join(out_dir, f"ebc_evolution_{EBC_WEIGHT_MODE}.mp4")

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=1.0 / max(float(FPS), 1.0))

    mp4_ok = True
    try:
        with imageio.get_writer(mp4_path, fps=FPS, codec="libx264", macro_block_size=1, ffmpeg_log_level="warning") as writer:
            for img in images:
                h, w = img.shape[:2]
                pad_h = h % 2
                pad_w = w % 2
                if pad_h or pad_w:
                    pad_width = ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w))
                    img = np.pad(img, pad_width, mode="edge")
                writer.append_data(img)
    except Exception as err:
        mp4_ok = False
        print(f"MP4 save failed: {err}")

    print("Saved frames to:")
    print(out_dir)
    print("Saved GIF:")
    print(gif_path)
    if mp4_ok:
        print("Saved MP4:")
        print(mp4_path)


if __name__ == "__main__":
    main()
