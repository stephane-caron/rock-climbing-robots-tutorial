# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for contact sets."""

import unittest

import numpy as np

from static_balance import ContactSet, FlatContact, PointContact


class TestContactSet(unittest.TestCase):
    """Tests for contact sets."""

    def setUp(self):
        """Prepare a set with a point and a flat contact."""
        self.point_contact = PointContact(
            [0.1, 0.2, 0.3], [0.0, -1.0, 1.0], 0.5, max_force=10.0
        )
        self.flat_contact = FlatContact(
            [-0.1, 0.0, 0.0], np.eye(3), 0.1, 0.05, 0.7
        )
        self.contact_set = ContactSet(
            contact for contact in (self.point_contact, self.flat_contact)
        )

    def test_len_and_iter(self):
        """Contacts given by a generator can be iterated more than once."""
        self.assertEqual(len(self.contact_set), 2)
        self.assertEqual(
            list(self.contact_set), [self.point_contact, self.flat_contact]
        )
        self.assertEqual(len(list(self.contact_set)), 2)

    def test_grasp_matrix_stacks_contacts(self):
        """Set grasp matrix stacks the grasp matrices of all contacts."""
        point = np.array([0.3, -0.2, 0.5])
        G = self.contact_set.compute_grasp_matrix(point)
        self.assertEqual(G.shape, (6, 3 + 6))
        self.assertTrue(
            np.allclose(
                G,
                np.hstack(
                    [
                        self.point_contact.compute_grasp_matrix(point),
                        self.flat_contact.compute_grasp_matrix(point),
                    ]
                ),
            )
        )

    def test_inequalities_are_block_diagonal(self):
        """Inequalities of each contact only involve its coordinates."""
        A, b = self.contact_set.wrench_inequalities
        A_point, b_point = self.point_contact.wrench_inequalities
        A_flat, b_flat = self.flat_contact.wrench_inequalities
        m = A_point.shape[0]
        self.assertEqual(A.shape, (m + A_flat.shape[0], 3 + 6))
        self.assertTrue(np.allclose(A[:m, :3], A_point))
        self.assertTrue(np.allclose(A[m:, 3:], A_flat))
        self.assertTrue(np.allclose(A[:m, 3:], 0.0))
        self.assertTrue(np.allclose(A[m:, :3], 0.0))
        self.assertTrue(np.allclose(b, np.hstack([b_point, b_flat])))

    def test_grasp_matrix_at_another_point(self):
        """Moving the reference point shifts moments by a cross product."""
        point = np.array([1.0, 2.0, 3.0])
        G_origin = self.contact_set.compute_grasp_matrix(np.zeros(3))
        G_point = self.contact_set.compute_grasp_matrix(point)
        forces = G_origin[:3]
        self.assertTrue(np.allclose(G_point[:3], forces))
        self.assertTrue(
            np.allclose(
                G_point[3:],
                G_origin[3:] - np.cross(point, forces, axis=0),
            )
        )
