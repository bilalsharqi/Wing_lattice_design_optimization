#=================================================================================
#1 - imports
#=================================================================================

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from scipy.sparse.linalg import eigsh

#=================================================================================
#2 - functions
#=================================================================================

def expand_network_to_3d(verts, edges, width=0.5):
    """
    Expands a 2D network into a 3D network by replacing each vertex 
    with a tetrahedron and connecting them based on the original edges.
    
    Parameters:
    - verts: (N, 2) array-like of 2D vertex positions.
    - edges: (E, 2) array-like of vertex indices defining connections.
    - width: float, scale factor for the tetrahedra.
    
    Returns:
    - subpoints: (4N, 3) array of the new 3D vertex positions.
    - subedges: (M, 2) array of the new edge connections by index.
    """
    verts = np.asarray(verts)
    edges = np.asarray(edges)
    
    # Define the tetrahedron basis
    basis_points = width * np.array([
        [1, 1, 1], 
        [1, -1, -1], 
        [-1, 1, -1], 
        [-1, -1, 1]
    ])
    
    num_verts = len(verts)
    
    # Lift 2D vertices to 3D (z=0) and expand into tetrahedra
    verts_3d = np.hstack([verts, np.zeros((num_verts, 1))])
    subpoints = (verts_3d[:, np.newaxis, :] + basis_points).reshape(-1, 3)
    
    # Construct intra-tetrahedral network
    intra_edges_template = np.array([[0, 3], [0, 2], [0, 1], [1, 3], [1, 2], [2, 3]])
    all_intra_edges = np.vstack([
        intra_edges_template + 4 * i for i in range(num_verts)
    ])

    # Add inter-tetrahedral edges
    subedges_list = []
    
    for edge in edges:
        u, v = edge.astype(int)
        
        # Connect corresponding basis points across original edges
        for j in range(4):
            subedges_list.append([4 * u + j, 4 * v + j])

        # Connect across original edges using the intra-edge pattern
        for j in range(len(intra_edges_template)):
            idx1, idx2 = intra_edges_template[j]
            subedges_list.append([4 * u + idx1, 4 * v + idx2])

    # Combine all edges
    subedges = np.vstack([np.array(subedges_list), all_intra_edges])
    
    return subpoints, subedges


def generate_3d_spring_constants(num_2d_verts, num_2d_edges, k_2d_list=None, k_intra=1.0):
    """
    Generates a list of spring constants for the expanded 3D network.
    
    Parameters:
    - num_2d_verts: int, number of vertices in the original 2D network.
    - num_2d_edges: int, number of edges in the original 2D network.
    - k_2d_list: array-like, spring constants for the original 2D edges. 
                 Defaults to an array of 1.0s.
    - k_intra: float, the stiff spring constant for intra-tetrahedral edges. Defaults to 1000.0.
    
    Returns:
    - k_3d_list: 1D numpy array of spring constants for all 3D subedges.
    """
    # Handle the default case for the 2D spring constants
    if k_2d_list is None:
        k_2d_list = 1*np.ones(num_2d_edges)
    else:
        k_2d_list = np.asarray(k_2d_list)
        
    # Block 1: 4 straight connections per original 2D edge
    # np.repeat duplicates each element N times (e.g., [k0, k0, k0, k0, k1, k1, k1, k1...])
    k_inter_straight = np.repeat(k_2d_list, 4)
    
    # Block 2: 6 cross connections per original 2D edge
    k_inter_cross = np.repeat(k_2d_list, 6)
    
    # Block 3: 6 intra-connections per original 2D vertex
    # np.full creates an array of the specified size filled with k_intra
    k_intra_edges = np.full(6 * num_2d_verts, k_intra)
    
    # Combine them in the exact order the subedges were created in expand_network_to_3d
    k_3d_list = np.concatenate([k_inter_straight, k_inter_cross, k_intra_edges])
    
    return k_3d_list

def pinnedCmat(vertex_positions, edgesByIndex, Cartesian_direction=0, smallest_boolean=True, frac_selected=0.05):
    """
    Constructs a pinned compatibility matrix, and returns the reduced geometry.
    Pins a specified fraction of vertices based on their spatial ranking.
    
    Parameters:
    - vertex_positions: (N, 3) array-like of vertex coordinates.
    - edgesByIndex: (E, 2) array-like of edge connections (0-indexed).
    - Cartesian_direction: int (0 for X, 1 for Y, 2 for Z).
    - smallest_boolean: bool, True to pin lowest coords, False to pin highest.
    - frac_selected: float, the fraction of total vertices to pin (e.g., 0.1 for 10%).
    
    Returns:
    - pinned_cMat: Sparse compatibility matrix with pinned columns removed.
    - unpinned_vertex_positions: (N_unpinned, 3) array of unpinned vertex coordinates.
    - unpinned_edges: (E_unpinned, 2) array of edges, re-indexed to the unpinned vertices.
    """
    vertex_positions = np.asarray(vertex_positions)
    edgesByIndex = np.asarray(edgesByIndex)
    
    num_verts = len(vertex_positions)
    num_edges = len(edgesByIndex)
    
    # 1. Edges by endpoints
    p1 = vertex_positions[edgesByIndex[:, 0]]
    p2 = vertex_positions[edgesByIndex[:, 1]]
    
    # 2. Construct normal cMat
    diff = p1 - p2
    lengths = np.linalg.norm(diff, axis=1, keepdims=True)
    hats = diff / lengths
    
    rows = np.repeat(np.arange(num_edges), 6)
    cols = np.zeros(num_edges * 6, dtype=int)
    data = np.zeros(num_edges * 6)
    
    u = edgesByIndex[:, 0]
    v = edgesByIndex[:, 1]
    
    cols[0::6] = 3 * u;      cols[1::6] = 3 * u + 1;  cols[2::6] = 3 * u + 2
    cols[3::6] = 3 * v;      cols[4::6] = 3 * v + 1;  cols[5::6] = 3 * v + 2
    
    data[0::6] = hats[:, 0]; data[1::6] = hats[:, 1]; data[2::6] = hats[:, 2]
    data[3::6] = -hats[:, 0];data[4::6] = -hats[:, 1];data[5::6] = -hats[:, 2]
    
    cMat = csr_matrix((data, (rows, cols)), shape=(num_edges, 3 * num_verts))
    
    # 3. Identify pinned and unpinned vertices based on fraction
    coords = vertex_positions[:, Cartesian_direction]
    
    # Determine the number of vertices to pin based on the fraction
    num_to_pin = int(round(num_verts * frac_selected))
    
    # Sort vertices by their coordinate along the chosen axis
    sorted_indices = np.argsort(coords)
    
    # Slice the sorted array to get the lowest or highest fraction
    if smallest_boolean:
        pinned_verts = sorted_indices[:num_to_pin]
    else:
        pinned_verts = sorted_indices[-num_to_pin:]
    
    # Get unpinned vertices
    all_verts = np.arange(num_verts)
    unpinned_vert_indices = np.setdiff1d(all_verts, pinned_verts)
    unpinned_vertex_positions = vertex_positions[unpinned_vert_indices]
    
    # 4. Construct pinned_cMat
    pinned_dofs = []
    for pv in pinned_verts:
        pinned_dofs.extend([3 * pv, 3 * pv + 1, 3 * pv + 2])
        
    mask = np.ones(3 * num_verts, dtype=bool)
    mask[pinned_dofs] = False
    remaining_dofs = np.where(mask)[0]
    pinned_cMat = cMat[:, remaining_dofs]
    
    # 5. Filter and Re-index Edges
    old_to_new_map = np.full(num_verts, -1, dtype=int)
    old_to_new_map[unpinned_vert_indices] = np.arange(len(unpinned_vert_indices))
    
    valid_edges_mask = (old_to_new_map[u] != -1) & (old_to_new_map[v] != -1)
    valid_edges_old = edgesByIndex[valid_edges_mask]
    
    unpinned_edges = old_to_new_map[valid_edges_old]
    
    return pinned_cMat, unpinned_vertex_positions, unpinned_edges, valid_edges_mask


def get_imposed_displacements(
    unpinned_vertex_positions, 
    selection_cartesian_direction=0, 
    selection_cartesian_smallest_boolean=False, 
    selection_fraction=0.05, 
    displacement_vector=np.array([0,0,10])
):
    """
    Selects a fraction of boundary vertices and assigns a displacement vector to them.
    
    Parameters:
    - unpinned_vertex_positions: (N, 3) array of vertex coordinates.
    - selection_cartesian_direction: int (0 for X, 1 for Y, 2 for Z).
    - selection_cartesian_smallest_boolean: bool, True for lowest coords, False for highest.
    - selection_fraction: float, fraction of vertices to select (e.g., 0.1 for 10%).
    - displacement_vector: array-like of 3 components (e.g., [0.5, 0.0, 0.0]).
    
    Returns:
    - v_list_dof: 1D array of the exact matrix DOF indices for the selected vertices.
    - u_list_flat: 1D array of the corresponding displacement values.
    - selected_verts: 1D array of the raw vertex indices (for plotting/reference).
    """
    positions = np.asarray(unpinned_vertex_positions)
    disp_vec = np.asarray(displacement_vector)
    
    num_verts = len(positions)
    num_to_select = int(round(num_verts * selection_fraction))
    
    # Extract the coordinates along the chosen axis
    coords = positions[:, selection_cartesian_direction]
    
    # Sort vertices by their coordinate
    sorted_indices = np.argsort(coords)
    
    # Select the lowest or highest fraction of vertices
    if selection_cartesian_smallest_boolean:
        selected_verts = sorted_indices[:num_to_select]
    else:
        selected_verts = sorted_indices[-num_to_select:]
        
    # --- Convert to matrix Degrees of Freedom (DOFs) ---
    # Each vertex 'i' corresponds to DOFs: 3i (X), 3i+1 (Y), 3i+2 (Z)
    v_list_dof = []
    for v in selected_verts:
        v_list_dof.extend([3 * v, 3 * v + 1, 3 * v + 2])
        
    v_list_dof = np.array(v_list_dof)
    
    # Repeat the displacement vector for each selected vertex
    # np.tile copies the [x, y, z] block 'num_to_select' times into a flat 1D array
    u_list_flat = np.tile(disp_vec, num_to_select)
    
    return v_list_dof, u_list_flat, selected_verts

def response_to_static_displacements(Cmat, k_list, v_list, u_list):
    """
    Cmat: Sparse matrix (e.g., incidence matrix)
    k_list: List of stiffness values for edges
    v_list: Indices of vertices with imposed displacements (0-indexed)
    u_list: Values of the imposed displacements
    """
    # Ensure Cmat is in a sparse format for efficient math
    Cmat = (csr_matrix(Cmat)).T #used to not take .T
    n_total = Cmat.shape[0]
    
    # Admittance matrix: Diagonal of 1/k
    # Using sparse diagonal matrix for memory efficiency
    admittance_mat = diags(1.0 / np.array(k_list), format='csr')
    
    d_mat = Cmat @ admittance_mat @ Cmat.T
    
    # Identify indices for 'a' (fixed) and 'b' (free) sets
    # Note: Python uses 0-based indexing
    v_list = np.array(v_list)
    vb_list = np.setdiff1d(np.arange(n_total), v_list)
    
    # Submatrix construction using efficient slicing
    # Daa = d_mat[v_list][:, v_list]
    daa = d_mat[v_list[:, None], v_list]
    # Dba = d_mat[vb_list][:, v_list]
    dba = d_mat[vb_list[:, None], v_list]
    # Dbb = d_mat[vb_list][:, vb_list]
    dbb = d_mat[vb_list[:, None], vb_list]
    
    # Imposed displacements (ua)
    ua = np.array(u_list)
    
    # Solve for response displacements (uB)
    # Solve Dbb * uB = -Dba * ua
    p = -dba @ ua
    u_b = spsolve(dbb, p)
    
    # Calculate forces (fA)
    dab = dba.T
    f_a = (daa @ ua) + (dab @ u_b)
    f_a_3d = f_a.reshape(-1, 3)
    total_applied_force = np.sum(f_a_3d, axis=0)
    
    # Combine all displacements into a single vector
    u = np.zeros(n_total)
    u[v_list] = ua
    u[vb_list] = u_b
    
    # Calculate tensions/currents (t)
    t = admittance_mat @ Cmat.T @ u
    
    return u, t, total_applied_force


def calculate_vibrational_modes(Cmat, k_list, k):
    """
    Calculates the k lowest-frequency vibrational modes of the network.
    
    Parameters:
    - Cmat: Original compatibility matrix (edges x DOFs)
    - k_list: List of spring constants (admittances)
    - k: Integer, number of lowest-frequency modes to find
    
    Returns:
    - frequencies: 1D array of the k lowest frequencies.
    - eigenvectors_list: List of 1D arrays (the mode shapes).
    - tensions_list: List of 1D arrays (tensions for each mode).
    """
    Cmat = csr_matrix(Cmat).T
    
    # Construct Admittance and Dynamical matrices
    admittance_mat = diags([1.0 / np.array(k_list)], [0], format='csr')
    d_mat = Cmat @ admittance_mat @ Cmat.T
    
    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(d_mat, k=k, sigma=-1e-6, which='LM')
    eigenvalues = np.maximum(eigenvalues, 0)
    frequencies = np.sqrt(eigenvalues)
    
    # Calculate tensions
    tensions_matrix = Cmat.T @ eigenvectors
    eigenvectors_list = [eigenvectors[:, i] for i in range(k)]
    tensions_list = [tensions_matrix[:, i] for i in range(k)]
    
    return np.array(frequencies), np.array(eigenvectors_list), np.array(tensions_list)


#=================================================================================
#3 - execution
#=================================================================================

#Choosing which tool you want, response to a load, or eigenmode calculation
response_to_load_boolean = True
plotted_mode_index = 1

#importing data
nodes_df = pd.read_csv('nodes_AP.csv')
edges_df = pd.read_csv('edges_AP.csv')
verts = nodes_df.to_numpy()
edges = edges_df.to_numpy()

#removing the long edges included in the original data **REMOVE THIS SECTION FOR OTHER NETWORKS**
indices_to_remove = [22, 232]
indices_to_remove = [i for i in indices_to_remove if i < len(edges)]
edges = np.delete(edges, indices_to_remove, axis=0)

#making edges/vertices in 3d as tetrahedral units
num_2d_verts = len(verts)
num_2d_edges = len(edges)
verts, edges = expand_network_to_3d(verts, edges)
verts_3d = verts
k_3d_list = generate_3d_spring_constants(num_2d_verts, num_2d_edges)

#constructing the compatibility matrix
Cmat, verts, edges, valid_edges_mask = pinnedCmat(verts, edges)

#defining variables used later outside of the if statement
active_tensions = np.zeros(1)
all_displacements = np.zeros(1)


if response_to_load_boolean:
    #applying load to network
    v_list_dof, u_list_flat, selected_verts = get_imposed_displacements(verts)
    all_displacements, all_tensions, total_applied_force = response_to_static_displacements(Cmat, k_3d_list, v_list_dof, u_list_flat)
    active_tensions = all_tensions[valid_edges_mask]
    print("Total applied force:", total_applied_force)
else:
    #finding eigenmodes
    frequencies, modes, tensions = calculate_vibrational_modes(Cmat, k_3d_list, plotted_mode_index+1)
    all_displacements = modes[plotted_mode_index]*10/np.max(modes[plotted_mode_index])
    all_tensions = tensions[plotted_mode_index]
    active_tensions = all_tensions[valid_edges_mask]
    print("Mode frequency:", frequencies[plotted_mode_index])


#=================================================================================
#4 - plotting
#=================================================================================

# 1. Setup the 3D figure
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})

# --- VERTICES ---

# Plot the background (pinned & unpinned) vertices as faint black dots
ax.scatter(verts_3d[:, 0], verts_3d[:, 1], verts_3d[:, 2], 
           color='black', s=10, alpha=0.3, zorder=1, label='All Vertices (Background)')

# Extract coordinates for the active unpinned vertices using 'verts'
x_unp = verts[:, 0]
y_unp = verts[:, 1]
z_unp = verts[:, 2]

# Plot the active unpinned vertices darker to stand out
ax.scatter(x_unp, y_unp, z_unp, color='black', s=15, alpha=0.9, zorder=4)

# --- EDGES (TENSION VISUALIZATION) ---

# Construct line segments for the active network using 'verts' and 'edges'
segments = verts[edges]

# Identify zero and non-zero tensions 
tol = 1e-8
zero_mask = np.abs(active_tensions) < tol
nonzero_mask = ~zero_mask

# Add zero-tension edges (Dashed, thin)
if np.any(zero_mask):
    lc_zero = Line3DCollection(segments[zero_mask], colors='gray', 
                               linewidths=0.5, linestyles='dashed', alpha=0.5, zorder=2)
    ax.add_collection3d(lc_zero)

# Add non-zero-tension edges (Solid, thickness scaled by magnitude)
if np.any(nonzero_mask):
    max_t = np.max(np.abs(active_tensions[nonzero_mask]))
    normalized_tensions = np.abs(active_tensions[nonzero_mask]) / (max_t + 1e-12)
    dynamic_linewidths = 0.5 + (4.0 * normalized_tensions)
    
    lc_nonzero = Line3DCollection(segments[nonzero_mask], colors='black', 
                                  linewidths=dynamic_linewidths, alpha=0.8, zorder=3)
    ax.add_collection3d(lc_nonzero)

# --- DISPLACEMENT ARROWS ---

# Reshape the 1D displacement array into an (N, 3) matrix
disp_3d = all_displacements.reshape(-1, 3)
u_vec, v_vec, w_vec = disp_3d[:, 0], disp_3d[:, 1], disp_3d[:, 2]

# Calculate magnitudes for color mapping
disp_magnitudes = np.linalg.norm(disp_3d, axis=1)

# Create a Blue -> Red colormap 
cmap = plt.get_cmap('turbo')
norm = mcolors.Normalize(vmin=disp_magnitudes.min(), vmax=disp_magnitudes.max())
arrow_colors = cmap(norm(disp_magnitudes))

# Filter out vertices that have practically zero displacement
moved_mask = disp_magnitudes > 1e-10

if np.any(moved_mask):
    ax.quiver(x_unp[moved_mask], y_unp[moved_mask], z_unp[moved_mask], 
              u_vec[moved_mask], v_vec[moved_mask], w_vec[moved_mask], 
              colors=arrow_colors[moved_mask], length=1.0, normalize=False, 
              arrow_length_ratio=0.3, zorder=5)

# --- COLORBAR ---
# Create a scalar mappable to link the colormap to the numerical values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Displacement Magnitude')

# --- AESTHETICS & SCALING ---

ax.set_title("")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Auto-scale the limits based on ALL vertices (so the background fits perfectly)
buffer = 0.1
ax.set_xlim(verts_3d[:, 0].min() - buffer, verts_3d[:, 0].max() + buffer)
ax.set_ylim(verts_3d[:, 1].min() - buffer, verts_3d[:, 1].max() + buffer)
ax.set_zlim(verts_3d[:, 2].min() - buffer, verts_3d[:, 2].max() + buffer)

# Aspect ratio bounding box
x_range = verts_3d[:, 0].max() - verts_3d[:, 0].min()
y_range = verts_3d[:, 1].max() - verts_3d[:, 1].min()
z_range = verts_3d[:, 2].max() - verts_3d[:, 2].min()
ax.set_box_aspect([x_range, y_range, z_range])

plt.tight_layout()
plt.show()










