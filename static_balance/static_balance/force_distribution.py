# SPDX-License-Identifier: GPL-3.0-or-later

"""Distribution of contact forces in static equilibrium."""

import numpy as np
from qpsolvers import solve_qp

from .contact_set import ContactSet
from .static_equilibrium import GRAVITY


def compute_supporting_forces(
    contact_set: ContactSet,
    com: np.ndarray,
    mass: float,
    solver: str = "daqp",
) -> list[np.ndarray] | None:
    r"""Compute contact forces that hold the robot in static equilibrium.

    Contact coordinates :math:`x` hold the robot when their net wrench
    compensates gravity, :math:`G(p_G) x = (0, 0, m g, 0, 0, 0)` with
    :math:`G(p_G)` the grasp matrix at the center of mass, and when they
    satisfy contact inequalities :math:`A x \leq b`. There are usually many
    such solutions: we pick the one with minimum norm :math:`\|x\|_2` by
    solving a quadratic program.

    Args:
        contact_set: Set of contacts.
        com: Position of the center of mass in the world frame.
        mass: Mass of the robot.
        solver: Quadratic programming solver, see `qpsolvers
            <https://github.com/qpsolvers/qpsolvers>`_.

    Returns:
        Contact coordinates of each contact, in its contact frame (forces for
        point contacts, wrenches at the center of the rectangle for flat
        contacts), or None if the robot cannot be in static equilibrium.
    """
    G = contact_set.compute_grasp_matrix(np.asarray(com, dtype=float))
    A, b = contact_set.wrench_inequalities
    num_coordinates = G.shape[1]
    gravity_compensation = np.array([0.0, 0.0, mass * GRAVITY, 0.0, 0.0, 0.0])
    x = solve_qp(
        P=np.eye(num_coordinates),
        q=np.zeros(num_coordinates),
        G=A,
        h=b,
        A=G,
        b=gravity_compensation,
        solver=solver,
    )
    if x is None:
        return None
    sizes = [contact.wrench_map.shape[1] for contact in contact_set]
    return np.split(x, np.cumsum(sizes)[:-1])
