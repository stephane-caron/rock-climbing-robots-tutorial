# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper functions for the 3D rotation group."""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    r"""Skew-symmetric matrix of a 3D vector.

    Args:
        v: 3D vector.

    Returns:
        Matrix :math:`[v]_\times` such that :math:`[v]_\times u = v \times
        u` for any 3D vector :math:`u`.
    """
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def rotation_from_normal(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix of a frame whose z-axis is along a given normal.

    Args:
        normal: Normal vector, not necessarily of unit norm.

    Returns:
        Rotation matrix from the frame to the world frame. The orientation of
        its x- and y-axes around the normal is unspecified.
    """
    z = np.array(normal, dtype=float) / np.linalg.norm(normal)
    other = np.array([1.0, 0.0, 0.0])
    if abs(z[0]) > 0.9:
        other = np.array([0.0, 1.0, 0.0])
    x = np.cross(other, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
