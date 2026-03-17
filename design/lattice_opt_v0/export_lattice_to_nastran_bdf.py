import os
import math
import json
import h5py
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

# Input optimizer H5
H5_FILE = r"results\Voronoi\voronoi_area_ON_EBC_veto_OFF_prune_mode_stress\beam\data\lattice_opt_beam_voronoi.h5"

# Which iteration to export: "initial", "final", or integer
ITERATION = "final"

# Output BDF file
OUTPUT_BDF = r"results\Voronoi\voronoi_area_ON_EBC_veto_OFF_prune_mode_stress\beam\data\lattice_opt_beam_voronoi.bdf"

# Cross-section control
USE_H5_AINIT = True
OVERRIDE_AREA = 3.175e-5   # used only if USE_H5_AINIT = False

# Minimum geometric tolerances used to suppress degenerate beam segments
# that can trigger FEMAP/Nastran colinear-orientation warnings.
MIN_BEAM_LENGTH = 1e-8         # meters; skip any beam shorter than this
MIN_ORIENT_NORM = 1e-14        # robustness tolerance for orientation construction

# Beam meshing / subdivision
TARGET_ELEMENT_LENGTH = 0.03
ELEMENT_COUNT_ROUNDING = "round"   # "round", "ceil", "floor"

# Material
YOUNGS_MODULUS = 70.0e9
POISSON_RATIO = 0.33
DENSITY = 2700.0

# Boundary conditions
ROOT_AXIS = 1
ROOT_TOL = None
SPC_ID = 1

# Element / property cards
MID = 1
PID = 1
ELEMENT_TYPE = "CBEAM"   # "CBEAM" or "CBAR"

# Optional active loads
WRITE_ACTIVE_LOADS = False
WRITE_ACTIVE_MOMENTS = False
LOAD_ID = 1
FORCE_SCALE = 1.0
MOMENT_SCALE = 1.0

# Also include extracted H5 loads as comments
WRITE_LOAD_COMMENTS = True


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


def root_boundary_nodes(nodes, axis=1, tol=None):
    vals = nodes[:, int(axis)]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = max(vmax - vmin, 1e-12)
    if tol is None:
        tol = max(1e-9, 1e-6 * span)
    return np.where(np.abs(vals - vmin) <= float(tol))[0]


def circular_props_from_area(area):
    area = float(area)
    if area <= 0.0:
        raise ValueError("Area must be positive")
    r = math.sqrt(area / math.pi)
    d = 2.0 * r
    I = math.pi * r**4 / 4.0
    J = math.pi * r**4 / 2.0
    return d, area, I, J


def choose_orientation_vector(x1, x2):
    """
    Return a robust, non-axis-aligned vector perpendicular to the beam axis.
    This avoids FEMAP/Nastran complaints for some perfectly vertical or
    perfectly horizontal CBEAM segments.
    """
    axis = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
    L = np.linalg.norm(axis)
    if L <= MIN_BEAM_LENGTH:
        return np.array([0.70710678, 0.70710678, 0.0], dtype=float)

    ex = axis / L

    # Use skew reference vectors instead of global basis directions
    refs = [
        np.array([1.0, 1.0, 1.0], dtype=float),
        np.array([1.0, -1.0, 2.0], dtype=float),
        np.array([2.0, 1.0, -1.0], dtype=float),
    ]

    for ref in refs:
        ref = ref / np.linalg.norm(ref)
        v = np.cross(ex, ref)
        nv = np.linalg.norm(v)
        if nv > MIN_ORIENT_NORM:
            return v / nv

    # Final fallback
    return np.array([0.70710678, 0.70710678, 0.0], dtype=float)


def edge_num_elements(length, target_length, rounding_mode):
    if target_length <= 0.0:
        raise ValueError("TARGET_ELEMENT_LENGTH must be positive")
    ratio = float(length) / float(target_length)
    rounding_mode = str(rounding_mode).lower()
    if rounding_mode == "round":
        n = int(np.round(ratio))
    elif rounding_mode == "ceil":
        n = int(np.ceil(ratio))
    elif rounding_mode == "floor":
        n = int(np.floor(ratio))
    else:
        raise ValueError("ELEMENT_COUNT_ROUNDING must be 'round', 'ceil', or 'floor'")
    return max(1, n)


def subdivide_edge(p0, p1, n_elem):
    if n_elem <= 1:
        return np.zeros((0, 3), dtype=float)
    pts = []
    for k in range(1, n_elem):
        a = k / float(n_elem)
        pts.append((1.0 - a) * p0 + a * p1)
    return np.asarray(pts, dtype=float)


def read_settings_json(h5):
    meta = h5.get("meta", None)
    if meta is None:
        raise RuntimeError("No /meta group found in H5")
    if "settings_json" not in meta.attrs:
        raise RuntimeError("No meta attribute 'settings_json' found in H5")
    raw = meta.attrs["settings_json"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def extract_a_init(settings):
    if isinstance(settings, dict):
        if "optimization" in settings and isinstance(settings["optimization"], dict):
            opt = settings["optimization"]
            if "a_init" in opt:
                return float(opt["a_init"])
        if "a_init" in settings:
            return float(settings["a_init"])
    raise RuntimeError("Could not find a_init in settings_json")


def ff_line(*fields):
    out = []
    for f in fields:
        if f is None:
            out.append("")
        elif isinstance(f, (int, np.integer)):
            out.append(str(int(f)))
        elif isinstance(f, str):
            out.append(f)
        else:
            out.append(f"{float(f):.12g}")
    return ",".join(out) + "\n"


def write_vector_card(f, card_name, sid, nid, vec, scale=1.0):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    mag = float(np.linalg.norm(vec))
    if mag <= 0.0:
        return
    n = vec / mag
    f.write(ff_line(card_name, sid, nid, None, scale * mag, n[0], n[1], n[2]))


def export_lattice_to_nastran(
    h5_file,
    output_bdf,
    iteration="final",
    target_element_length=0.03,
    element_count_rounding="round",
    use_h5_ainit=True,
    override_area=3.175e-5,
    youngs_modulus=70.0e9,
    poisson_ratio=0.33,
    density=2700.0,
    root_axis=1,
    root_tol=None,
    spc_id=1,
    mid=1,
    pid=1,
    element_type="CBEAM",
    write_active_loads=True,
    write_active_moments=True,
    load_id=1,
    force_scale=1.0,
    moment_scale=1.0,
    write_load_comments=True,
):
    with h5py.File(h5_file, "r") as h5:
        key = resolve_iter_key(h5, iteration)
        grp = h5[key]

        nodes = np.asarray(grp["nodes"][()], dtype=float)
        edges = np.asarray(grp["edges"][()], dtype=int)
        force_vector = np.asarray(grp["force_vector"][()]).reshape(-1) if "force_vector" in grp else None

        settings = read_settings_json(h5)
        h5_a_init = extract_a_init(settings)

    member_area = float(h5_a_init) if use_h5_ainit else float(override_area)
    member_diameter, member_area, I1, J = circular_props_from_area(member_area)
    I2 = I1

    root_nodes = root_boundary_nodes(nodes, axis=root_axis, tol=root_tol)

    nodal_loads = None
    dof_per_node = None
    if force_vector is not None and force_vector.size % nodes.shape[0] == 0:
        dof_per_node = force_vector.size // nodes.shape[0]
        nodal_loads = force_vector.reshape((nodes.shape[0], dof_per_node))

    fem_nodes = nodes.tolist()
    fem_elems = []
    n_skipped_subdivision = 0

    for i, j in edges:
        p0 = nodes[int(i)]
        p1 = nodes[int(j)]
        L = float(np.linalg.norm(p1 - p0))
        if L <= MIN_BEAM_LENGTH:
            n_skipped_subdivision += 1
            continue

        n_elem = edge_num_elements(L, target_element_length, element_count_rounding)
        internal_pts = subdivide_edge(p0, p1, n_elem)

        chain = [int(i)]
        for pt in internal_pts:
            fem_nodes.append(pt.tolist())
            chain.append(len(fem_nodes) - 1)
        chain.append(int(j))

        # First protection: skip tiny segments during subdivision
        for a, b in zip(chain[:-1], chain[1:]):
            pa = np.asarray(fem_nodes[a], dtype=float)
            pb = np.asarray(fem_nodes[b], dtype=float)
            Lseg = float(np.linalg.norm(pb - pa))
            if Lseg <= MIN_BEAM_LENGTH:
                n_skipped_subdivision += 1
                continue
            fem_elems.append({"ga": a, "gb": b})

    fem_nodes = np.asarray(fem_nodes, dtype=float)

    n_skipped_export = 0
    written_elem_count = 0

    with open(output_bdf, "w", encoding="utf-8") as f:
        f.write("SOL 101\n")
        f.write("CEND\n")
        f.write("TITLE = Lattice beam export from optimizer H5\n")
        f.write("SUBCASE 1\n")
        f.write(f"    SPC = {spc_id}\n")
        if write_active_loads or write_active_moments:
            f.write(f"    LOAD = {load_id}\n")
        f.write("BEGIN BULK\n")

        f.write("$ ------------------------------------------------------------\n")
        f.write(f"$ Source H5               : {h5_file}\n")
        f.write(f"$ Iteration               : {key}\n")
        f.write(f"$ Element type            : {element_type}\n")
        f.write(f"$ USE_H5_AINIT            : {use_h5_ainit}\n")
        f.write(f"$ H5 a_init               : {h5_a_init:.9e}\n")
        f.write(f"$ OVERRIDE_AREA           : {override_area:.9e}\n")
        f.write(f"$ Used beam area          : {member_area:.9e}\n")
        f.write(f"$ Equivalent diameter     : {member_diameter:.9e} m\n")
        f.write(f"$ TARGET_ELEMENT_LENGTH   : {target_element_length} m\n")
        f.write(f"$ ELEMENT_COUNT_ROUNDING  : {element_count_rounding}\n")
        f.write(f"$ MIN_BEAM_LENGTH         : {MIN_BEAM_LENGTH:.3e} m\n")
        f.write(f"$ Original nodes          : {nodes.shape[0]}\n")
        f.write(f"$ Original edges          : {edges.shape[0]}\n")
        f.write(f"$ FEM nodes               : {fem_nodes.shape[0]}\n")
        f.write(f"$ FEM beam elements       : {len(fem_elems)}\n")
        f.write(f"$ Root SPC nodes          : {len(root_nodes)}\n")
        f.write(f"$ WRITE_ACTIVE_LOADS      : {write_active_loads}\n")
        f.write(f"$ WRITE_ACTIVE_MOMENTS    : {write_active_moments}\n")
        f.write("$ ------------------------------------------------------------\n")

        f.write(ff_line("MAT1", mid, youngs_modulus, None, poisson_ratio, density))

        if element_type.upper() == "CBEAM":
            f.write(ff_line("PBEAM", pid, mid, member_area, I1, I2, None, J))
        elif element_type.upper() == "CBAR":
            f.write(ff_line("PBAR", pid, mid, member_area, I1, I2, J))
        else:
            raise ValueError("element_type must be 'CBEAM' or 'CBAR'")

        for nid, xyz in enumerate(fem_nodes, start=1):
            f.write(ff_line("GRID", nid, None, xyz[0], xyz[1], xyz[2]))

        # Second protection: skip tiny / degenerate segments again at write time
        for elem in fem_elems:
            ga = int(elem["ga"]) + 1
            gb = int(elem["gb"]) + 1
            pa = np.asarray(fem_nodes[ga - 1], dtype=float)
            pb = np.asarray(fem_nodes[gb - 1], dtype=float)

            Lseg = float(np.linalg.norm(pb - pa))
            if Lseg <= MIN_BEAM_LENGTH:
                n_skipped_export += 1
                continue

            x = choose_orientation_vector(pa, pb)
            written_elem_count += 1

            if element_type.upper() == "CBEAM":
                f.write(ff_line("CBEAM", written_elem_count, pid, ga, gb, x[0], x[1], x[2]))
            else:
                f.write(ff_line("CBAR", written_elem_count, pid, ga, gb, x[0], x[1], x[2]))

        root_ids = [int(i) + 1 for i in root_nodes.tolist()]
        max_ids_per_line = 8
        if root_ids:
            first = True
            for i in range(0, len(root_ids), max_ids_per_line):
                chunk = root_ids[i:i + max_ids_per_line]
                if first:
                    f.write(ff_line("SPC1", spc_id, "123456", *chunk))
                    first = False
                else:
                    f.write(ff_line("+", *chunk))

        if nodal_loads is not None:
            for nid in range(nodes.shape[0]):
                row = np.asarray(nodal_loads[nid], dtype=float).reshape(-1)
                if write_active_loads and len(row) >= 3:
                    write_vector_card(f, "FORCE", load_id, nid + 1, row[:3], scale=force_scale)
                if write_active_moments and len(row) >= 6:
                    write_vector_card(f, "MOMENT", load_id, nid + 1, row[3:6], scale=moment_scale)

        if write_load_comments and nodal_loads is not None:
            f.write("$ ------------------------------------------------------------\n")
            f.write("$ Extracted nodal loads from H5\n")
            f.write(f"$ dof_per_node = {dof_per_node}\n")
            f.write("$ original_node_id, Fx, Fy, Fz, [Mx, My, Mz if present]\n")
            for nid in range(nodes.shape[0]):
                row = nodal_loads[nid]
                if np.any(np.abs(row) > 0.0):
                    vals = ", ".join(f"{float(v):.6g}" for v in row)
                    f.write(f"$ {nid+1}, {vals}\n")
            f.write("$ ------------------------------------------------------------\n")

        f.write("ENDDATA\n")

    return {
        "output_bdf": output_bdf,
        "iteration": key,
        "original_nodes": int(nodes.shape[0]),
        "original_edges": int(edges.shape[0]),
        "fem_nodes": int(fem_nodes.shape[0]),
        "fem_beam_elements": int(written_elem_count),
        "used_area": float(member_area),
        "equivalent_diameter": float(member_diameter),
        "target_element_length": float(target_element_length),
        "element_rounding_mode": str(element_count_rounding),
        "n_skipped_subdivision": int(n_skipped_subdivision),
        "n_skipped_export": int(n_skipped_export),
    }


def main():
    info = export_lattice_to_nastran(
        h5_file=H5_FILE,
        output_bdf=OUTPUT_BDF,
        iteration=ITERATION,
        target_element_length=TARGET_ELEMENT_LENGTH,
        element_count_rounding=ELEMENT_COUNT_ROUNDING,
        use_h5_ainit=USE_H5_AINIT,
        override_area=OVERRIDE_AREA,
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        density=DENSITY,
        root_axis=ROOT_AXIS,
        root_tol=ROOT_TOL,
        spc_id=SPC_ID,
        mid=MID,
        pid=PID,
        element_type=ELEMENT_TYPE,
        write_active_loads=WRITE_ACTIVE_LOADS,
        write_active_moments=WRITE_ACTIVE_MOMENTS,
        load_id=LOAD_ID,
        force_scale=FORCE_SCALE,
        moment_scale=MOMENT_SCALE,
        write_load_comments=WRITE_LOAD_COMMENTS,
    )

    print("Wrote BDF:")
    print(info["output_bdf"])
    print(f'Iteration: {info["iteration"]}')
    print(f'Original nodes: {info["original_nodes"]}')
    print(f'Original edges: {info["original_edges"]}')
    print(f'FEM nodes: {info["fem_nodes"]}')
    print(f'FEM beam elements: {info["fem_beam_elements"]}')
    print(f'Used area: {info["used_area"]:.9e}')
    print(f'Equivalent diameter: {info["equivalent_diameter"]:.9e} m')
    print(f'Target element length: {info["target_element_length"]} m')
    print(f'Element rounding mode: {info["element_rounding_mode"]}')
    print(f'Skipped short segments during subdivision: {info["n_skipped_subdivision"]}')
    print(f'Skipped short segments during export: {info["n_skipped_export"]}')


if __name__ == "__main__":
    main()
