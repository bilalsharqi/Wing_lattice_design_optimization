from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
from collections import defaultdict
import pyvista as pv
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def parse_load_titles(bdf_path):
    """
    Parse BDF and print only the load set titles (comments starting with $)
    for each LOAD ID.
    """
    model = BDF(debug = False)
    model.read_bdf(bdf_path)

    print("\n--- LOAD SET TITLES ---")
    for load_id, load_cards in model.loads.items():
        # Each LOAD object has a "comment" attribute which contains the $ line
        # Sometimes it's a list, sometimes a string
        comment = load_cards[0].comment if hasattr(load_cards[0], 'comment') else None
        if comment:
            print(f"Load ID {int(load_id)}: {comment.strip()}")
        else:
            print(f"Load ID {int(load_id)}: (no comment)")

# ============================================================
# 1️⃣ Extract nodal stress
# ============================================================
def extract_nodal_stress(op2_path, subcase=2):
    op2 = OP2(debug=False)
    op2.read_op2(op2_path)
    stress = op2.op2_results.stress.ctetra_stress[subcase]

    elem_node = stress.element_node  # (N, 2): [element_id, node_id]
    data = stress.data  # shape: (ntimes, N, n_components)
    von_mises = data[0, :, 9]  # index 9 = von Mises stress

    # Average stresses per node
    node_stress_sum = defaultdict(float)
    node_stress_count = defaultdict(int)

    for (eid, nid), vm in zip(elem_node, von_mises):
        if nid == 0:  # skip centroids
            continue
        node_stress_sum[nid] += vm
        node_stress_count[nid] += 1

    node_stress = {nid: node_stress_sum[nid] / node_stress_count[nid]
                   for nid in node_stress_sum.keys()}

    return node_stress

def load_op2_results(op2_path):
    """Load OP2 file and return the CTETRA stress results object."""
    op2 = OP2(debug = False)
    op2.read_op2(op2_path)
    # print("Available results:\n", op2.op2_results)

    stress_obj = op2.op2_results.stress.ctetra_stress
    if stress_obj is None:
        raise ValueError("No CTETRA (tetrahedral) stress results found in this OP2 file.")
    return stress_obj


def extract_centroid_von_mises(stress_obj, subcase_id=1, time_index=0):
    """
    Extract von Mises stresses at element centroids.
    Returns (element_ids, von_mises_centroid).
    """
    stress = stress_obj[subcase_id]
    data = stress.data  # shape: (ntimes, n_records, nresults)
    element_node = stress.element_node  # (n_records, 2)

    # Mask for centroid entries only (node=0)
    centroid_mask = (element_node[:, 1] == 0)

    # Extract element IDs and von Mises stress (index=9)
    elem_ids = element_node[centroid_mask, 0]
    von_mises = data[time_index, centroid_mask, 9]

    return elem_ids, von_mises, stress

def compute_element_centroids(bdf_path, element_ids):
    """
    Load BDF file and compute centroids for given element IDs.
    Returns np.array of centroids (N, 3).
    """
    bdf = BDF(debug = False)
    bdf.read_bdf(bdf_path)

    centroids = []
    for eid in element_ids:
        elem = bdf.elements[eid]
        centroids.append(elem.Centroid())
    return np.array(centroids)

# ============================================================
# 2️⃣ Build PyVista mesh
# ============================================================
def load_bdf_mesh(bdf_path):
    model = BDF(debug=False)
    model.read_bdf(bdf_path)

    node_ids = np.array(sorted(model.nodes.keys()))
    coords = np.array([model.nodes[nid].get_position() for nid in node_ids])

    cells = []
    cell_types = []
    for eid, elem in model.elements.items():
        try:
            e_nodes = elem.node_ids
            node_indices = [np.where(node_ids == nid)[0][0] for nid in e_nodes]

            cells.append([len(node_indices)] + node_indices)

            etype = elem.type
            if etype in ("CTETRA", "TETRA"):
                cell_types.append(pv.CellType.TETRA)
            elif etype in ("CHEXA", "HEX8"):
                cell_types.append(pv.CellType.HEXAHEDRON)
            elif etype in ("CPENTA", "PENTA6"):
                cell_types.append(pv.CellType.WEDGE)
            elif etype in ("CPYRAM", "PYRAM5"):
                cell_types.append(pv.CellType.PYRAMID)
            else:
                continue
        except Exception:
            continue

    cells_flat = np.hstack(cells)
    cell_types = np.array(cell_types, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells_flat, cell_types, coords)
    return mesh, node_ids, coords

# ============================================================
# 3️⃣ Overlay stress onto PyVista mesh
# ============================================================
def overlay_stress_on_mesh_stable(mesh, node_ids, node_stress):
    # Build array in correct order of node_ids
    stress_array = np.array([node_stress.get(nid, np.nan) for nid in node_ids])
    mesh.point_data["Von Mises Stress"] = stress_array
    return mesh

def overlay_stress_on_mesh(mesh, node_ids, node_stress, default=np.nan):
    """
    Overlay nodal Von Mises stress on the full mesh.
    Missing nodes get `default` (NaN or 0).
    """
    stress_array = np.full(mesh.n_points, default, dtype=float)
    for i, nid in enumerate(node_ids):
        if nid in node_stress:
            stress_array[i] = node_stress[nid]
    mesh.point_data["Von Mises Stress"] = stress_array
    return mesh

# ============================================================
# 4️⃣ Plot stress field with overlay
# ============================================================
def plot_stress_overlay(mesh, cmap='turbo', show_nodes=False):
    """
    Plot stress overlay on mesh, automatically masking NaNs and using percentile color scaling.
    """
    stress_array = mesh.point_data['Von Mises Stress']
    stress_array = np.nan_to_num(stress_array, nan=0.0)
    mesh.point_data['Von Mises Stress'] = stress_array
    # Use percentiles for color scaling
    # vmin, vmax = np.percentile(stress_array, [0, 100])

    scalar_bar_args = {
        'title': 'Von Mises [MPa]',
        'vertical': True,
        'height': 0.82,       # full viewport height
        # 'width': 0.05,       # thin bar
        'position_x': 0.82,  # slightly right of the mesh (tweakable)
        'position_y': 0.05,   # align bottom with viewport
        'title_font_size': 24,
        'label_font_size': 18,
        # 'outline': True,
    }

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Von Mises Stress",
        cmap=cmap,
        show_edges=True,
        opacity=1.0,
        scalar_bar_args=scalar_bar_args
        # clim=[vmin, vmax]   # key: set color limits explicitly
    )


    plotter.add_mesh(mesh, color="black", style="wireframe", line_width=0.5, opacity=0.2)

    if show_nodes:
        plotter.add_points(mesh.points, color="red", point_size=6, render_points_as_spheres=True)

    plotter.add_axes()
    plotter.show_bounds(
        grid='front',
        location='outer',
        xtitle="X[m]",
        ytitle="Y[m]",
        ztitle="Z[m]",
        fmt="%.2f",
    )
    plotter.view_xy()
    plotter.show(title="Stress Overlay on BDF Mesh")


def plot_von_mises_centroids_plotly(centroids, von_mises):
    centroids = np.asarray(centroids)
    von_mises = np.asarray(von_mises)
    vmin, vmax = np.percentile(von_mises, [0, 100])
    vmin, vmax = np.min(von_mises), np.max(von_mises)
    colors = np.clip(von_mises, vmin, vmax)
    fig = go.Figure(data=[
        go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='turbo',
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title='Von Mises'),
                showscale=True,
                opacity=0.8
            )
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=0, y=0, z=3),       # Look along +Z at XY
                up=dict(x=0, y=1, z=0)
            )
        ),
        title='Von Mises Stress (per centroid)',
        template='plotly_white'                # Or remove for default
    )
    fig.show()


def plot_stress_distribution(centroids, von_mises):
    """3D scatter plot of von Mises stress at element centroids."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90) # look from the front

    sc = ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c=von_mises, cmap="viridis"
    )

    fig.colorbar(sc, label="Von Mises Stress")
    ax.set_title("Von Mises Stress per Element Centroid")
    ax.set_xlabel("X[m]")
    ax.set_ylabel("Y[m]")
    ax.set_zlabel("Z[m]")
    plt.tight_layout()
    # plt.show()

# ============================================================
# 5️⃣ Main pipeline
# ============================================================
if __name__ == "__main__":
    # op2_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\triangulated_lattice\V3\Triangulated_FEM\10_31_25_New\model-0000.op2"
    # bdf_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\triangulated_lattice\V3\Triangulated_FEM\10_31_25_New\MODEL-0000.bdf"
    # op2_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\hexagonal_lattice\V3\Hex_FEM\hex-0001.op2"
    # bdf_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\hexagonal_lattice\V3\Hex_FEM\Hex-0001.dat"
    # bdf_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\delaunay\V2\FEM\delaunay_v2-0001.dat"
    # op2_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\delaunay\V2\FEM\delaunay_v2-0001.op2"
    bdf_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\hud\V2\Hyperuniform_FEM\hyperuniform-0000.dat"
    op2_path = r"G:\.shortcut-targets-by-id\1yW0o_J_4L-8AIlBz0AvzsLEy0ZXlmdyj\COMPASS_Cesnik_Kotov\NetworkTesting\hud\V2\Hyperuniform_FEM\hyperuniform-0000.op2"
    # parse_load_titles(bdf_path)
    subcase_id = 2 # compressive

    node_stress = extract_nodal_stress(op2_path, subcase=subcase_id)
    mesh, node_ids, coords = load_bdf_mesh(bdf_path)
    mesh = overlay_stress_on_mesh(mesh, node_ids, node_stress, default = 0.0)
    plot_stress_overlay(mesh, show_nodes=False)


    # centroid analysis if needed ##
    # stress_obj = load_op2_results(op2_path)
    # elem_ids, von_mises_centroid, stress = extract_centroid_von_mises(stress_obj,subcase_id = subcase_id)
    # centroids = compute_element_centroids(bdf_path, elem_ids)
    # plot_von_mises_centroids_plotly(centroids, von_mises_centroid)

