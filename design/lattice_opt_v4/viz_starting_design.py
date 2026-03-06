import numpy as np
import matplotlib.pyplot as plt


def edge_lengths(nodes, edges):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    return np.linalg.norm(q - p, axis=1)


def classify_edges_by_direction(nodes, edges, tol=0.85):
    """
    Classify edges by dominant direction cosine.
    Returns masks for chordwise (x), spanwise (y), vertical (z), diagonal.
    """
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


def plot_edges_3d(ax, nodes, edges, mask=None, stride=1, color="k", alpha=0.5, lw=0.6):
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
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=color, alpha=alpha, linewidth=lw)

    ax.set_xlabel("x (chord)")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z (depth)")


def plot_edges_2d(ax, nodes, edges, plane="xy", mask=None, stride=1, color="k", alpha=0.4, lw=0.5):
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


def visualize_starting_design(nodes, edges, wing, slice_stations=(0.25, 0.5, 0.75)):
    lengths = edge_lengths(nodes, edges)
    chordwise, spanwise, vertical, diagonal = classify_edges_by_direction(nodes, edges)

    print("Starting design geometry summary:")
    print(f"  nodes                 = {len(nodes)}")
    print(f"  edges                 = {len(edges)}")
    print(f"  min / mean / max L    = {lengths.min():.4f}, {lengths.mean():.4f}, {lengths.max():.4f} m")
    print(f"  chordwise edges       = {np.sum(chordwise)}")
    print(f"  spanwise edges        = {np.sum(spanwise)}")
    print(f"  vertical edges        = {np.sum(vertical)}")
    print(f"  diagonal edges        = {np.sum(diagonal)}")

    # Figure 1: 3D categorized views
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(221, projection="3d")
    plot_edges_3d(ax1, nodes, edges, mask=spanwise, stride=3, color="C0", alpha=0.8, lw=0.8)
    ax1.set_title("Spanwise-dominant members")

    ax2 = fig.add_subplot(222, projection="3d")
    plot_edges_3d(ax2, nodes, edges, mask=chordwise, stride=3, color="C1", alpha=0.8, lw=0.8)
    ax2.set_title("Chordwise-dominant members")

    ax3 = fig.add_subplot(223, projection="3d")
    plot_edges_3d(ax3, nodes, edges, mask=vertical, stride=2, color="C2", alpha=0.8, lw=0.8)
    ax3.set_title("Vertical members")

    ax4 = fig.add_subplot(224, projection="3d")
    plot_edges_3d(ax4, nodes, edges, mask=diagonal, stride=6, color="0.2", alpha=0.35, lw=0.6)
    ax4.set_title("Diagonal members")

    plt.tight_layout()
    plt.show()

    # Figure 2: orthographic projections
    fig2, axs = plt.subplots(1, 3, figsize=(16, 5))
    plot_edges_2d(axs[0], nodes, edges, plane="xy", stride=6, color="k", alpha=0.35, lw=0.5)
    axs[0].set_title("Top view (x-y)")

    plot_edges_2d(axs[1], nodes, edges, plane="yz", stride=6, color="k", alpha=0.35, lw=0.5)
    axs[1].set_title("Front/span-depth view (y-z)")

    plot_edges_2d(axs[2], nodes, edges, plane="xz", stride=4, color="k", alpha=0.35, lw=0.5)
    axs[2].set_title("Chord-depth view (x-z)")

    plt.tight_layout()
    plt.show()

    # Figure 3: spanwise slice views
    fig3, axs = plt.subplots(1, len(slice_stations), figsize=(5 * len(slice_stations), 5))
    if len(slice_stations) == 1:
        axs = [axs]

    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]

    for ax, frac in zip(axs, slice_stations):
        y0 = frac * wing.span
        band = 0.08 * wing.span  # 8% span band
        mask = (
            (np.abs(p[:, 1] - y0) <= band) |
            (np.abs(q[:, 1] - y0) <= band)
        )

        for i, j in edges[mask]:
            pi = nodes[i]
            pj = nodes[j]
            ax.plot([pi[0], pj[0]], [pi[2], pj[2]], color="k", alpha=0.5, linewidth=0.7)

        ax.set_title(f"Slice near y/b = {frac:.2f}")
        ax.set_xlabel("x (chord)")
        ax.set_ylabel("z (depth)")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()