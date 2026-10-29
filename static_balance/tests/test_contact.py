# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for contacts."""

import unittest

import numpy as np

from static_balance import FlatContact, PointContact
from static_balance.contact import force_inequalities


def is_feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
    """Check whether a vector satisfies linear inequalities."""
    return bool(np.all(A @ x <= b + 1e-10))


class TestContact(unittest.TestCase):
    """Tests for point and flat contacts."""

    def test_friction_pyramid_is_inscribed(self):
        """Pyramid edges lie on the friction cone, and inside is feasible."""
        mu = 0.7
        A, b = force_inequalities(mu, np.inf)
        edge = np.array([mu / np.sqrt(2.0), mu / np.sqrt(2.0), 1.0])
        self.assertAlmostEqual(np.linalg.norm(edge[:2]), mu)
        self.assertTrue(is_feasible(A, b, 3.0 * edge))
        self.assertFalse(is_feasible(A, b, edge + [0.01, 0.0, 0.0]))
        self.assertFalse(is_feasible(A, b, np.array([0.0, 0.0, -1.0])))

    def test_max_force_is_conservative(self):
        """Feasible forces have a norm at most the maximum force."""
        mu, max_force = 0.7, 10.0
        A, b = force_inequalities(mu, max_force)
        self.assertTrue(is_feasible(A, b, np.array([0.0, 0.0, max_force])))
        self.assertFalse(
            is_feasible(A, b, np.array([0.0, 0.0, 1.01 * max_force]))
        )
        rng = np.random.default_rng(42)
        for force in rng.uniform([-5, -5, 0], [5, 5, 10], size=(1000, 3)):
            if is_feasible(A, b, force):
                self.assertLessEqual(np.linalg.norm(force), max_force)

    def test_infinite_friction_is_unilaterality(self):
        """Without friction limit, only the normal force sign matters."""
        A, b = force_inequalities(np.inf, np.inf)
        self.assertTrue(is_feasible(A, b, np.array([1e6, -1e6, 1e-6])))
        self.assertFalse(is_feasible(A, b, np.array([0.0, 0.0, -1e-3])))

    def test_point_contact_has_no_moment(self):
        """Point contacts map their coordinates to forces only."""
        contact = PointContact([0.1, 0.2, 0.3], [0.0, -1.0, 1.0], 0.5)
        self.assertEqual(contact.wrench_map.shape, (6, 3))
        self.assertTrue(np.allclose(contact.wrench_map[3:], 0.0))

    def test_grasp_matrix_at_point(self):
        """Forces rotate to the world frame, moments are cross products."""
        position = np.array([0.1, -0.2, 0.3])
        point = np.array([1.0, 2.0, 3.0])
        contact = PointContact(position, [0.0, -1.0, 1.0], 0.5)
        G = contact.compute_grasp_matrix(point)
        local_force = np.array([0.1, -0.2, 1.0])
        force, moment = G[:3] @ local_force, G[3:] @ local_force
        self.assertTrue(np.allclose(force, contact.rotation @ local_force))
        self.assertTrue(np.allclose(moment, np.cross(position - point, force)))

    def test_flat_contact_center_of_pressure(self):
        """Flat contacts accept centers of pressure inside their rectangle."""
        X, Y = 0.1, 0.05
        contact = FlatContact(np.zeros(3), np.eye(3), X, Y, 0.6)
        A, b = contact.wrench_inequalities
        self.assertEqual(contact.wrench_map.shape, (6, 6))
        for x, y, inside in [
            (0.0, 0.0, True),
            (0.09, -0.04, True),
            (0.11, 0.0, False),
            (0.0, -0.06, False),
        ]:
            wrench = np.array([0.0, 0.0, 1.0, y, -x, 0.0])
            self.assertEqual(is_feasible(A, b, wrench), inside)

    def test_flat_contact_max_force(self):
        """Flat contacts bound their resultant force."""
        contact = FlatContact(np.zeros(3), np.eye(3), 0.1, 0.05, 0.6, 5.0)
        A, b = contact.wrench_inequalities
        wrench = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        self.assertTrue(is_feasible(A, b, wrench))
        self.assertFalse(is_feasible(A, b, 1.01 * wrench))

    def test_invalid_arguments(self):
        """Invalid shapes, negative friction or force limits raise errors."""
        with self.assertRaises(ValueError):
            PointContact([0.0, 0.0], [0.0, 0.0, 1.0], 0.5)
        with self.assertRaises(ValueError):
            PointContact(np.zeros(3), [0.0, 0.0, 1.0], -0.1)
        with self.assertRaises(ValueError):
            PointContact(np.zeros(3), [0.0, 0.0, 1.0], 0.5, max_force=0.0)
