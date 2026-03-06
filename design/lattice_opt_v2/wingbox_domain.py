
import numpy as np


class WingBox:
    """
    Rectangular wingbox volume:
      x: chordwise [0, chord]
      y: spanwise  [0, span]   (root at y=0)
      z: thickness [-depth/2, +depth/2]
    Root clamp = all nodes with y <= root_tol
    """

    def __init__(self, span=4.0, chord=1.0, depth=0.16, root_tol=1e-6):
        self.span = float(span)
        self.chord = float(chord)
        self.depth = float(depth)
        self.root_tol = float(root_tol)

    def contains(self, pts_xyz: np.ndarray) -> np.ndarray:
        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        z = pts_xyz[:, 2]
        return (
            (x >= 0.0) & (x <= self.chord) &
            (y >= 0.0) & (y <= self.span) &
            (z >= -self.depth / 2.0) & (z <= self.depth / 2.0)
        )

    def root_mask(self, pts_xyz: np.ndarray) -> np.ndarray:
        return pts_xyz[:, 1] <= self.root_tol
