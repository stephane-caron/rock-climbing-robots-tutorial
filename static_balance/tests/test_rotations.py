# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for rotations."""

import unittest

import numpy as np

from static_balance.rotations import rotation_from_normal, skew


class TestRotations(unittest.TestCase):
    """Tests for rotations."""

    def test_skew(self):
        """Skew-symmetric matrices compute cross products."""
        u, v = np.array([1.0, 2.0, 3.0]), np.array([-4.0, 0.5, 2.0])
        self.assertTrue(np.allclose(skew(u) @ v, np.cross(u, v)))

    def test_rotation_from_normal(self):
        """Rotations from normals are proper with the normal as z-axis."""
        for normal in ([0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.3, -1.0, 0.5]):
            R = rotation_from_normal(normal)
            self.assertTrue(np.allclose(R.T @ R, np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(R), 1.0)
            unit = np.array(normal) / np.linalg.norm(normal)
            self.assertTrue(np.allclose(R[:, 2], unit))
