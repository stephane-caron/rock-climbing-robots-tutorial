# SPDX-License-Identifier: GPL-3.0-or-later

"""Static-equilibrium polygon of the center of mass."""

import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from .contact_set import ContactSet

GRAVITY = 9.81  # [m] / [s]²


def compute_static_equilibrium_polygon(
    contact_set: ContactSet,
    mass: float,
) -> np.ndarray:
    """Compute the static-equilibrium polygon of a contact set.

    The robot is in static equilibrium when contact wrenches compensate
    gravity. This is feasible if and only if the horizontal coordinates of
    its center of mass lie in the static-equilibrium polygon, regardless of
    the height of its center of mass. The world frame is assumed to have its
    z-axis pointing up, opposite to gravity.

    The polygon is computed by projecting the polytope of feasible contact
    coordinates onto the horizontal plane, using the incremental projection
    algorithm from Bretl and Lall (2008).

    Args:
        contact_set: Set of contacts.
        mass: Mass of the robot. The polygon only depends on it when some
            contacts have a maximum force.

    Returns:
        Array of shape (n, 2) of polygon vertices in counterclockwise order.
    """
    G = contact_set.compute_grasp_matrix(np.zeros(3))
    A, b = contact_set.wrench_inequalities

    # The net contact wrench at the origin is G @ x. In static equilibrium,
    # it opposes gravity: the force is (0, 0, mg) and the moment is
    # p_G x (0, 0, mg) = mg * (y_G, -x_G, 0).
    mg = mass * GRAVITY
    E = np.vstack([-G[4], G[3]]) / mg
    f = np.zeros(2)
    C = G[[0, 1, 2, 5]]
    d = np.array([0.0, 0.0, mg, 0.0])

    vertices = pypoman.project_polytope(
        proj=(E, f),
        ineq=(A, b),
        eq=(C, d),
        method="bretl",
    )
    # Bretl's algorithm yields clockwise vertices, sometimes traced twice
    # when its initial triangle is clockwise. The convex hull removes
    # duplicates and orders vertices counterclockwise.
    points = np.array(vertices)
    return points[ConvexHull(points).vertices]
