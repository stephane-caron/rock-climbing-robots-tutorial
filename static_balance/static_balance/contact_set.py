# SPDX-License-Identifier: GPL-3.0-or-later

"""Sets of contacts acting on the same robot."""

from typing import Iterable, Iterator

import numpy as np
from scipy.linalg import block_diag

from .contact import Contact


class ContactSet:
    """Set of contacts acting on the same robot.

    Contacts can be of any type. Their coordinates stack into a vector
    :math:`x`, mapped to the net contact wrench by the grasp matrix of the
    set, and constrained by the inequalities of all contacts.

    Attributes:
        contacts: List of contacts.
    """

    contacts: list[Contact]

    def __init__(self, contacts: Iterable[Contact]):
        """Initialize contact set.

        Args:
            contacts: Contacts in the set.
        """
        self.contacts = list(contacts)

    def __iter__(self) -> Iterator[Contact]:
        """Iterate over contacts in the set."""
        return iter(self.contacts)

    def __len__(self) -> int:
        """Number of contacts in the set."""
        return len(self.contacts)

    def compute_grasp_matrix(self, point: np.ndarray) -> np.ndarray:
        """Map from stacked contact coordinates to the net contact wrench.

        Args:
            point: Point where the net wrench is taken, in the world frame.

        Returns:
            Matrix of shape (6, n), with n the total number of contact
            coordinates, mapping them to the net contact wrench. The wrench
            stacks a force and a moment taken at the given point, both in the
            world frame.
        """
        return np.hstack(
            [contact.compute_grasp_matrix(point) for contact in self.contacts]
        )

    @property
    def wrench_inequalities(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Inequalities on stacked contact coordinates.

        Returns:
            Pair :math:`(A, b)` such that feasible stacked contact coordinates
            :math:`x` satisfy :math:`A x \leq b`.
        """
        inequalities = [contact.wrench_inequalities for contact in self]
        A = block_diag(*[A_i for A_i, _ in inequalities])
        b = np.hstack([b_i for _, b_i in inequalities])
        return A, b
