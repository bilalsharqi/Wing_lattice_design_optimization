import numpy as np


def classify_square_lattice_edges(nodes, edges, tol=0.90):
    """
    Classify edges by dominant direction:
      - spanwise (y)
      - chordwise (x)
      - vertical (z)
      - diagonal (everything else)
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


def edge_midpoints(nodes, edges):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    return 0.5 * (p + q)


def build_spanwise_bay_groups(nodes, edges, wing, n_span_bins=14, tol=1e-10):
    """
    Build grouped pruning candidates for diagonal members only.

    New behavior:
      - split diagonals by spanwise bin
      - split diagonals by orientation sign in each principal plane

    This makes groups much smaller and more local than the old
    diag_spanbin_* groups.
    """
    mids = edge_midpoints(nodes, edges)
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    d = q - p

    chordwise, spanwise, vertical, diagonal = classify_square_lattice_edges(nodes, edges)

    ymid = mids[:, 1]
    bins = np.linspace(0.0, wing.span, n_span_bins + 1)

    groups = []
    group_labels = []

    dx = d[:, 0]
    dy = d[:, 1]
    dz = d[:, 2]

    # classify diagonals by which plane they mainly live in
    xy_like = diagonal & (np.abs(dz) < tol)
    yz_like = diagonal & (np.abs(dx) < tol)
    xz_like = diagonal & (np.abs(dy) < tol)

    # orientation sign in each plane
    xy_pos = xy_like & ((dx * dy) > 0.0)
    xy_neg = xy_like & ((dx * dy) < 0.0)

    yz_pos = yz_like & ((dy * dz) > 0.0)
    yz_neg = yz_like & ((dy * dz) < 0.0)

    xz_pos = xz_like & ((dx * dz) > 0.0)
    xz_neg = xz_like & ((dx * dz) < 0.0)

    families = [
        ("xy_pos", xy_pos),
        ("xy_neg", xy_neg),
        ("yz_pos", yz_pos),
        ("yz_neg", yz_neg),
        ("xz_pos", xz_pos),
        ("xz_neg", xz_neg),
    ]

    for ib in range(n_span_bins):
        y0 = bins[ib]
        y1 = bins[ib + 1]
        in_bin = (ymid >= y0) & (ymid < y1)

        for name, fam_mask in families:
            mask = in_bin & fam_mask
            idx = np.where(mask)[0]
            if len(idx) > 0:
                groups.append(idx)
                group_labels.append(f"{name}_spanbin_{ib:02d}")

    return groups, group_labels


def evaluate_group_scores(edges, areas, member_force, member_stress, groups, sigma_allow):
    """
    Score each candidate group for pruning.
    Prefer groups that are:
      - low stress
      - low force
      - near minimum area
    Lower score = better pruning candidate
    """
    max_abs_force = max(float(np.max(np.abs(member_force))), 1e-16)
    min_area = float(np.min(areas))

    scores = []
    for gidx in groups:
        stress_util = np.mean(np.abs(member_stress[gidx]) / max(sigma_allow, 1e-16))
        force_util = np.mean(np.abs(member_force[gidx]) / max_abs_force)
        area_util = np.mean(areas[gidx] / max(min_area, 1e-16))

        score = 0.45 * stress_util + 0.45 * force_util + 0.10 * area_util
        scores.append(score)

    return np.asarray(scores)


def prune_one_group(edges, areas, group_edge_indices):
    keep = np.ones(len(edges), dtype=bool)
    keep[group_edge_indices] = False

    if np.sum(keep) == 0:
        return edges, areas, np.zeros((0,), dtype=int)

    return edges[keep], areas[keep], np.asarray(group_edge_indices, dtype=int)