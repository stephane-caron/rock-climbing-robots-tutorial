# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for static-equilibrium polygons."""

import unittest

import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from static_balance import (
    ContactSet,
    FlatContact,
    PointContact,
    compute_static_equilibrium_polygon,
)
from static_balance.static_equilibrium import GRAVITY

MASS = 2.0  # [kg]


def polygon_area(vertices: np.ndarray) -> float:
    """Area of a 2D polygon, positive for counterclockwise vertices."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


class TestStaticEquilibrium(unittest.TestCase):
    """Tests for static-equilibrium polygons."""

    def test_flat_ground_is_convex_hull(self):
        """On flat ground, the polygon is the hull of contact points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.2, 0.8, 0.0],
                [0.1, 1.0, 0.0],
            ]
        )
        contact_set = ContactSet(
            PointContact(p, [0.0, 0.0, 1.0], 0.5) for p in points
        )
        vertices = compute_static_equilibrium_polygon(contact_set, MASS)
        hull = ConvexHull(points[:, :2])
        self.assertGreater(polygon_area(vertices), 0.0)  # counterclockwise
        self.assertAlmostEqual(polygon_area(vertices), hull.volume, places=4)

    def test_flat_contact_is_its_rectangle(self):
        """A flat contact on the ground yields its rectangle."""
        contact = FlatContact([0.3, -0.2, 0.0], np.eye(3), 0.5, 0.25, 0.7)
        vertices = compute_static_equilibrium_polygon(
            ContactSet([contact]), MASS
        )
        self.assertAlmostEqual(polygon_area(vertices), 0.5, places=4)
        self.assertTrue(np.allclose(vertices.mean(axis=0), [0.3, -0.2]))

    def test_independent_of_height(self):
        """Lifting all contacts does not change the polygon."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [0.5, 1.0, 0.1]])
        normals = [[0.1, 0.0, 1.0], [-0.2, 0.1, 1.0], [0.0, -0.3, 1.0]]
        polygons = [
            compute_static_equilibrium_polygon(
                ContactSet(
                    PointContact(p + [0.0, 0.0, dz], n, 0.6)
                    for p, n in zip(points, normals)
                ),
                MASS,
            )
            for dz in (0.0, 1.0)
        ]
        areas = [polygon_area(polygon) for polygon in polygons]
        self.assertAlmostEqual(areas[0], areas[1], places=4)

    def square_contacts(self, max_force: float) -> ContactSet:
        """Point contacts at the corners of a unit square on the ground."""
        return ContactSet(
            PointContact([x, y, 0.0], [0.0, 0.0, 1.0], 0.5, max_force)
            for x, y in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        )

    def test_max_force_shrinks_polygon(self):
        """Contacts that cannot carry the whole weight shrink the polygon."""
        weight = MASS * GRAVITY
        unbounded = compute_static_equilibrium_polygon(
            self.square_contacts(np.inf), MASS
        )
        bounded = compute_static_equilibrium_polygon(
            self.square_contacts(0.5 * weight), MASS
        )
        self.assertAlmostEqual(polygon_area(unbounded), 1.0, places=4)
        self.assertGreater(polygon_area(bounded), 0.0)
        self.assertLess(polygon_area(bounded), 0.99)

    def test_max_force_depends_on_mass(self):
        """With force limits, a lighter robot has a larger polygon."""
        contact_set = self.square_contacts(0.5 * MASS * GRAVITY)
        heavy = compute_static_equilibrium_polygon(contact_set, MASS)
        light = compute_static_equilibrium_polygon(contact_set, 0.7 * MASS)
        self.assertGreater(polygon_area(light), polygon_area(heavy))

    def test_max_force_below_weight_is_infeasible(self):
        """Contacts that cannot carry the weight together have no polygon."""
        contact_set = self.square_contacts(0.2 * MASS * GRAVITY)
        with self.assertRaises(pypoman.Exception):
            compute_static_equilibrium_polygon(contact_set, MASS)
