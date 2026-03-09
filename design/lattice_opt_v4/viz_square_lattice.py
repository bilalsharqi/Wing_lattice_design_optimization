import numpy as np
import matplotlib.pyplot as plt


def classify_square_lattice_edges(nodes, edges, tol=0.90):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    d = q - p
    L = np.linalg.norm(d, axis=1, keepdims=True)
    n = d / np.maximum(L, 1e-16)

    ax = np.abs(n[:, 0])
    ay = np.abs(n[:, 1])
    az = np.abs(n[:, 2])

    chordwise = ax >= tol
    spanwise = ay >= tol
    vertical = az >= tol
    diagonal = ~(chordwise | spanwise | vertical)

    return chordwise, spanwise, vertical, diagonal


def plot_edges_3d(ax, nodes, edges, mask=None, stride=1, color="k", alpha=0.8, lw=0.8):
    if mask is None:
        use_edges = edges
    else:
        use_edges = edges[mask]

    if len(use_edges) == 0:
        return

    use = np.arange(0, len(use_edges), max(1, stride))
    for e in use:
        i, j = use_edges[e]
        p = nodes[i]
        q = nodes[j]
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=color,
            alpha=alpha,
            linewidth=lw,
        )

    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z (depth)")


def plot_edges_2d(ax, nodes, edges, plane="xy", mask=None, stride=1, color="k", alpha=0.6, lw=0.7):
    if mask is None:
        use_edges = edges
    else:
        use_edges = edges[mask]

    if len(use_edges) == 0:
        return

    use = np.arange(0, len(use_edges), max(1, stride))
    for e in use:
        i, j = use_edges[e]
        p = nodes[i]
        q = nodes[j]

        if plane == "xy":
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, alpha=alpha, linewidth=lw)
            ax.set_xlabel("x (chord)")
            ax.set_ylabel("y (span)")
        elif plane == "yz":
            ax.plot([p[1], q[1]], [p[2], q[2]], color=color, alpha=alpha, linewidth=lw)
            ax.set_xlabel("y (span)")
            ax.set_ylabel("z (depth)")
        elif plane == "xz":
            ax.plot([p[0], q[0]], [p[2], q[2]], color=color, alpha=alpha, linewidth=lw)
            ax.set_xlabel("x (chord)")
            ax.set_ylabel("z (depth)")
        else:
            raise ValueError("plane must be one of: xy, yz, xz")


def edge_lengths(nodes, edges):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    return np.linalg.norm(q - p, axis=1)


def visualize_square_lattice(nodes, edges, wing, slice_stations=(0.2, 0.5, 0.8)):
    chordwise, spanwise, vertical, diagonal = classify_square_lattice_edges(nodes, edges)
    lengths = edge_lengths(nodes, edges)

    print("Square lattice geometry summary:")
    print(f"  nodes                 = {len(nodes)}")
    print(f"  edges                 = {len(edges)}")
    print(f"  min / mean / max L    = {lengths.min():.4f}, {lengths.mean():.4f}, {lengths.max():.4f} m")
    print(f"  spanwise members      = {np.sum(spanwise)}")
    print(f"  chordwise members     = {np.sum(chordwise)}")
    print(f"  vertical members      = {np.sum(vertical)}")
    print(f"  diagonal members      = {np.sum(diagonal)}")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_edges_3d(ax, nodes, edges, spanwise, stride=1, color="C0", alpha=0.9, lw=1.0)
    plot_edges_3d(ax, nodes, edges, chordwise, stride=1, color="C1", alpha=0.9, lw=1.0)
    plot_edges_3d(ax, nodes, edges, vertical, stride=1, color="C2", alpha=0.9, lw=1.0)
    plot_edges_3d(ax, nodes, edges, diagonal, stride=1, color="0.4", alpha=0.35, lw=0.7)
    ax.set_title("Square wing-box lattice: colored by member family")
    plt.tight_layout()
    plt.show()

    fig2, axs = plt.subplots(1, 3, figsize=(16, 5))

    plot_edges_2d(axs[0], nodes, edges, plane="xy", mask=spanwise, stride=1, color="C0", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[0], nodes, edges, plane="xy", mask=chordwise, stride=1, color="C1", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[0], nodes, edges, plane="xy", mask=diagonal, stride=1, color="0.5", alpha=0.35, lw=0.7)
    axs[0].set_title("Top view (x-y)")

    plot_edges_2d(axs[1], nodes, edges, plane="yz", mask=spanwise, stride=1, color="C0", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[1], nodes, edges, plane="yz", mask=vertical, stride=1, color="C2", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[1], nodes, edges, plane="yz", mask=diagonal, stride=1, color="0.5", alpha=0.35, lw=0.7)
    axs[1].set_title("Span-depth view (y-z)")

    plot_edges_2d(axs[2], nodes, edges, plane="xz", mask=chordwise, stride=1, color="C1", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[2], nodes, edges, plane="xz", mask=vertical, stride=1, color="C2", alpha=0.9, lw=1.0)
    plot_edges_2d(axs[2], nodes, edges, plane="xz", mask=diagonal, stride=1, color="0.5", alpha=0.35, lw=0.7)
    axs[2].set_title("Chord-depth view (x-z)")

    # for ax in axs:
    #     ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def visualize_damage_on_square_lattice(nodes, edges, removed_node_indices, surviving_nodes=None, surviving_edges=None):
    chordwise, spanwise, vertical, diagonal = classify_square_lattice_edges(nodes, edges)

    fig = plt.figure(figsize=(18, 6))

    # Original with damage markers
    ax1 = fig.add_subplot(131, projection="3d")
    plot_edges_3d(ax1, nodes, edges, spanwise, stride=1, color="C0", alpha=0.20, lw=0.8)
    plot_edges_3d(ax1, nodes, edges, chordwise, stride=1, color="C1", alpha=0.20, lw=0.8)
    plot_edges_3d(ax1, nodes, edges, vertical, stride=1, color="C2", alpha=0.20, lw=0.8)
    plot_edges_3d(ax1, nodes, edges, diagonal, stride=1, color="0.5", alpha=0.12, lw=0.6)
    if len(removed_node_indices) > 0:
        pts = nodes[np.asarray(removed_node_indices, dtype=int)]
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", s=35, label="Removed nodes")
        ax1.legend()
    ax1.set_title("Original lattice + removed nodes")

    # Removed edges only
    ax2 = fig.add_subplot(132, projection="3d")
    if len(removed_node_indices) > 0:
        removed_set = set(np.asarray(removed_node_indices, dtype=int).tolist())
        removed_edge_mask = np.array([(i in removed_set) or (j in removed_set) for i, j in edges], dtype=bool)
        plot_edges_3d(ax2, nodes, edges, removed_edge_mask, stride=1, color="red", alpha=0.85, lw=1.0)
    ax2.set_title("Edges incident to removed nodes")

    # Surviving root-connected component
    ax3 = fig.add_subplot(133, projection="3d")
    if surviving_nodes is not None and surviving_edges is not None and len(surviving_edges) > 0:
        c2, s2, v2, d2 = classify_square_lattice_edges(surviving_nodes, surviving_edges)
        plot_edges_3d(ax3, surviving_nodes, surviving_edges, s2, stride=1, color="C0", alpha=0.9, lw=1.0)
        plot_edges_3d(ax3, surviving_nodes, surviving_edges, c2, stride=1, color="C1", alpha=0.9, lw=1.0)
        plot_edges_3d(ax3, surviving_nodes, surviving_edges, v2, stride=1, color="C2", alpha=0.9, lw=1.0)
        plot_edges_3d(ax3, surviving_nodes, surviving_edges, d2, stride=1, color="0.4", alpha=0.35, lw=0.7)
    ax3.set_title("Surviving root-connected damaged graph")

    plt.tight_layout()
    plt.show()