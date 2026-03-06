import numpy as np


class WingBox:
    def __init__(self, span=4.0, chord=1.0, depth=0.16, root_tol=1e-6):
        self.span = float(span)
        self.chord = float(chord)
        self.depth = float(depth)
        self.root_tol = float(root_tol)

    def root_mask(self, pts_xyz: np.ndarray) -> np.ndarray:
        return pts_xyz[:, 1] <= self.root_tol