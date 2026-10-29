# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the distribution of contact forces."""

import unittest

import numpy as np

from static_balance import (
    ContactSet,
    FlatContact,
    PointContact,
    compute_static_equilibrium_polygon,
    compute_supporting_forces,
)
from static_balance.static_equilibrium import GRAVITY

MASS = 2.0  # [kg]


def square_contacts(max_force: float = np.inf) -> ContactSet:
    """Point contacts at the corners of a unit square on the ground."""
    return ContactSet(
        PointContact([x, y, 0.0], [0.0, 0.0, 1.0], 0.5, max_force)
        for x, y in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )


def net_wrench(contact_set: ContactSet, forces, point) -> np.ndarray:
    """Net contact wrench at a point from contact coordinates."""
    G = contact_set.compute_grasp_matrix(np.asarray(point, dtype=float))
    return G @ np.hstack(forces)


class TestForceDistribution(unittest.TestCase):
    """Tests for the distribution of contact forces."""

    def test_symmetric_distribution(self):
        """A centered CoM is supported equally by symmetric contacts."""
        forces = compute_supporting_forces(
            square_contacts(), [0.5, 0.5, 0.8], MASS
        )
        self.assertEqual(len(forces), 4)
        for force in forces:
            self.assertTrue(
                np.allclose(force, [0.0, 0.0, MASS * GRAVITY / 4], atol=1e-6)
            )

    def test_static_equilibrium(self):
        """Supporting forces compensate gravity and satisfy inequalities."""
        contact_set = square_contacts(max_force=8.0)
        com = np.array([0.3, 0.6, 0.8])
        forces = compute_supporting_forces(contact_set, com, MASS)
        wrench = net_wrench(contact_set, forces, com)
        self.assertTrue(
            np.allclose(wrench, [0, 0, MASS * GRAVITY, 0, 0, 0], atol=1e-6)
        )
        for contact, force in zip(contact_set, forces):
            A, b = contact.wrench_inequalities
            self.assertTrue(np.all(A @ force <= b + 1e-6))

    def test_com_outside_support(self):
        """There are no supporting forces when the CoM is outside."""
        self.assertIsNone(
            compute_supporting_forces(square_contacts(), [1.5, 0.5, 0.8], MASS)
        )

    def test_max_force_below_weight(self):
        """There are no supporting forces when contacts are too weak."""
        contact_set = square_contacts(max_force=0.2 * MASS * GRAVITY)
        self.assertIsNone(
            compute_supporting_forces(contact_set, [0.5, 0.5, 0.8], MASS)
        )

    def test_consistent_with_polygon(self):
        """Forces exist inside the polygon and not outside."""
        contact_set = square_contacts(max_force=0.5 * MASS * GRAVITY)
        polygon = compute_static_equilibrium_polygon(contact_set, MASS)
        center = polygon.mean(axis=0)
        for vertex in polygon:
            inside = center + 0.95 * (vertex - center)
            outside = center + 1.05 * (vertex - center)
            self.assertIsNotNone(
                compute_supporting_forces(contact_set, [*inside, 0.8], MASS)
            )
            self.assertIsNone(
                compute_supporting_forces(contact_set, [*outside, 0.8], MASS)
            )

    def test_single_point_contact(self):
        """One point contact holds a CoM right above it, and nothing else."""
        contact_set = ContactSet(
            [PointContact([0.2, 0.1, 0.0], [0.0, 0.0, 1.0], 0.5)]
        )
        forces = compute_supporting_forces(contact_set, [0.2, 0.1, 0.8], MASS)
        self.assertTrue(
            np.allclose(forces[0], [0.0, 0.0, MASS * GRAVITY], atol=1e-6)
        )
        self.assertIsNone(
            compute_supporting_forces(contact_set, [0.3, 0.1, 0.8], MASS)
        )

    def test_flat_contact_coordinates(self):
        """Flat contacts return wrenches at the center of their rectangle."""
        contact = FlatContact([0.0, 0.0, 0.0], np.eye(3), 0.1, 0.05, 0.7)
        forces = compute_supporting_forces(
            ContactSet([contact]), [0.05, 0.0, 0.8], MASS
        )
        self.assertEqual(forces[0].shape, (6,))
        weight = MASS * GRAVITY
        self.assertTrue(
            np.allclose(forces[0], [0, 0, weight, 0, -0.05 * weight, 0])
        )
