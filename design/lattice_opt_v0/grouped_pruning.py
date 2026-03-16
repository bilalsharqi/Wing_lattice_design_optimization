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


def compute_edge_lengths(nodes, edges):
    p = nodes[edges[:, 0]]
    q = nodes[edges[:, 1]]
    return np.linalg.norm(q - p, axis=1)


def compute_edge_masses(nodes, edges, areas, density):
    return density * areas * compute_edge_lengths(nodes, edges)


def _safe_group_mean(vals, idx):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.inf
    return float(np.mean(vals[idx]))


def _safe_group_max(vals, idx):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.inf
    return float(np.max(vals[idx]))


def _safe_group_mass_fraction(edge_masses, idx, total_mass):
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0 or total_mass <= 0.0:
        return 0.0
    return float(np.sum(edge_masses[idx]) / total_mass)


def evaluate_group_scores(
    nodes,
    edges,
    areas,
    member_stress,
    sigma_allow,
    groups,
    density,
    prune_score_mode="hybrid",
    edge_ebc=None,
    ebc_backbone_veto=True,
    ebc_bcrit=0.80,
    w_sigma=0.50,
    w_ebc=0.40,
    w_mass=0.10,
    random_state=None,
):
    """
    Score each candidate group for pruning.

    Returns
    -------
    scores : ndarray
        Lower score means more pruneable. Vetoed groups are set to +inf.
    veto_mask : ndarray[bool]
        True for EBC-protected groups.
    metrics : dict
        Group-level diagnostics for HDF5/debugging.
    """
    n_groups = len(groups)
    if n_groups == 0:
        return (
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=bool),
            {
                "group_mean_util": np.zeros((0,), dtype=float),
                "group_mean_ebc": np.zeros((0,), dtype=float),
                "group_max_ebc": np.zeros((0,), dtype=float),
                "group_mass_fraction": np.zeros((0,), dtype=float),
            },
        )

    util = np.abs(np.asarray(member_stress).reshape(-1)) / max(float(sigma_allow), 1e-16)
    edge_masses = compute_edge_masses(nodes, edges, areas, density)
    total_mass = float(np.sum(edge_masses))

    if edge_ebc is None:
        edge_ebc = np.zeros((len(edges),), dtype=float)
    else:
        edge_ebc = np.asarray(edge_ebc, dtype=float).reshape(-1)

    group_mean_util = np.zeros((n_groups,), dtype=float)
    group_mean_ebc = np.zeros((n_groups,), dtype=float)
    group_max_ebc = np.zeros((n_groups,), dtype=float)
    group_mass_fraction = np.zeros((n_groups,), dtype=float)

    for gi, gidx in enumerate(groups):
        group_mean_util[gi] = _safe_group_mean(util, gidx)
        group_mean_ebc[gi] = _safe_group_mean(edge_ebc, gidx)
        group_max_ebc[gi] = _safe_group_max(edge_ebc, gidx)
        group_mass_fraction[gi] = _safe_group_mass_fraction(edge_masses, gidx, total_mass)

    veto_mask = np.zeros((n_groups,), dtype=bool)
    if ebc_backbone_veto:
        veto_mask = group_max_ebc > float(ebc_bcrit)

    mode = str(prune_score_mode).lower().strip()
    if mode == "random":
        rng = np.random.default_rng(random_state)
        scores = rng.random(n_groups)
    elif mode == "stress_only":
        scores = group_mean_util - 0.10 * group_mass_fraction
    elif mode == "ebc_only":
        scores = group_mean_ebc - 0.10 * group_mass_fraction
    elif mode == "hybrid":
        scores = (
            float(w_sigma) * group_mean_util
            + float(w_ebc) * group_mean_ebc
            - float(w_mass) * group_mass_fraction
        )
    else:
        raise ValueError(f"Unknown prune_score_mode='{prune_score_mode}'")

    scores = np.asarray(scores, dtype=float)
    scores[veto_mask] = np.inf

    metrics = {
        "group_mean_util": group_mean_util,
        "group_mean_ebc": group_mean_ebc,
        "group_max_ebc": group_max_ebc,
        "group_mass_fraction": group_mass_fraction,
        "group_veto_mask": veto_mask.astype(bool),
    }
    return scores, veto_mask, metrics


def prune_one_group(edges, areas, group_edge_indices):
    keep = np.ones(len(edges), dtype=bool)
    keep[group_edge_indices] = False

    if np.sum(keep) == 0:
        return edges, areas, np.zeros((0,), dtype=int)

    return edges[keep], areas[keep], np.asarray(group_edge_indices, dtype=int)
