# SPDX-License-Identifier: GPL-3.0-or-later

"""Contacts between a robot and its environment."""

from abc import ABC, abstractmethod

import numpy as np

from .rotations import rotation_from_normal, skew


def force_inequalities(
    friction: float, max_force: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Inequalities on a contact force in the frame of its contact normal.

    The force :math:`f` lies in a pyramid inscribed in the friction cone
    :math:`\|f_{xy}\| \leq \mu f_z`, which is a conservative approximation.
    Its norm is bounded by :math:`|f_x| + |f_y| + f_z \leq f_{\max}`, which is
    also conservative since :math:`\|f\|_2 \leq \|f\|_1`.

    Args:
        friction: Static friction coefficient :math:`\mu`.
        max_force: Maximum force :math:`f_{\max}`, infinite for no bound.

    Returns:
        Pair :math:`(A, b)` such that feasible forces satisfy :math:`A f \leq
        b`.
    """
    rows = [[0.0, 0.0, -1.0]]  # unilaterality
    if np.isfinite(friction):
        mu = friction / np.sqrt(2.0)  # inner approximation
        rows += [
            [-1.0, 0.0, -mu],
            [+1.0, 0.0, -mu],
            [0.0, -1.0, -mu],
            [0.0, +1.0, -mu],
        ]
    A = np.array(rows)
    b = np.zeros(len(rows))
    if np.isfinite(max_force):
        A = np.vstack([A, [[sx, sy, 1.0] for sx in (-1, 1) for sy in (-1, 1)]])
        b = np.hstack([b, np.full(4, max_force)])
    return A, b


class Contact(ABC):
    """Contact between a robot and its environment.

    A contact constrains some degrees of freedom of the motion of a robot body
    relative to its environment. In return, the environment applies a contact
    wrench on the robot along the constrained directions. Its coordinates lie
    in a polyhedron that accounts for unilaterality, friction and force
    limits. For instance, point contacts constrain the three translations and
    leave rotations free, while flat contacts constrain all six degrees of
    freedom. Other combinations, such as line contacts, can be added by
    implementing :meth:`wrench_map` and :meth:`wrench_inequalities`. See e.g.
    `Contact modes <https://scaron.info/robotics/contact-modes.html>`_ for
    further details.

    The contact frame has its origin at the contact point, and its z-axis
    along the contact normal, pointing from the environment to the robot.

    Attributes:
        position: Position of the contact point in the world frame.
        rotation: Rotation matrix from the contact frame to the world frame.
        friction: Static friction coefficient.
        max_force: Maximum norm of the contact force, infinite for no bound.
    """

    position: np.ndarray
    rotation: np.ndarray
    friction: float
    max_force: float

    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        friction: float,
        max_force: float = np.inf,
    ):
        """Initialize contact.

        Args:
            position: Position of the contact point in the world frame.
            rotation: Rotation matrix from the contact frame to the world
                frame.
            friction: Static friction coefficient.
            max_force: Maximum norm of the contact force, infinite for no
                bound.
        """
        position = np.array(position, dtype=float)
        rotation = np.array(rotation, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"position has shape {position.shape} != (3,)")
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation has shape {rotation.shape} != (3, 3)")
        if friction < 0.0:
            raise ValueError(f"friction {friction} is negative")
        if max_force <= 0.0:
            raise ValueError(f"max_force {max_force} is not positive")
        self.position = position
        self.rotation = rotation
        self.friction = friction
        self.max_force = max_force

    @property
    def normal(self) -> np.ndarray:
        """Contact normal in the world frame."""
        return self.rotation[:, 2]

    @property
    @abstractmethod
    def wrench_map(self) -> np.ndarray:
        """Map from contact coordinates to the contact wrench.

        Contact coordinates are the components of the contact wrench along
        the degrees of freedom that the contact constrains.

        Returns:
            Matrix of shape (6, n), with n the number of constrained degrees
            of freedom, mapping contact coordinates to the contact wrench. The
            wrench stacks a force and a moment taken at the contact point,
            both in the contact frame.
        """

    @property
    @abstractmethod
    def wrench_inequalities(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Inequalities on contact coordinates.

        Returns:
            Pair :math:`(A, b)` such that feasible contact coordinates
            :math:`x` satisfy :math:`A x \leq b`.
        """

    def compute_grasp_matrix(self, point: np.ndarray) -> np.ndarray:
        """Map from contact coordinates to the contact wrench at a point.

        Args:
            point: Point where the wrench is taken, in the world frame.

        Returns:
            Matrix of shape (6, n) mapping contact coordinates to the contact
            wrench. The wrench stacks a force and a moment taken at the given
            point, both in the world frame.
        """
        R = self.rotation
        adjoint = np.block(
            [
                [R, np.zeros((3, 3))],
                [skew(self.position - point) @ R, R],
            ]
        )
        return adjoint @ self.wrench_map


class PointContact(Contact):
    """Point contact.

    A point contact constrains the three translations of the contact point
    and leaves rotations free, for instance a ball in a socket. The
    environment can only apply a force at the contact point, in its friction
    cone, and no moment. Contact coordinates are the three components of the
    contact force in the contact frame.
    """

    def __init__(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        friction: float,
        max_force: float = np.inf,
    ):
        """Initialize point contact.

        Args:
            position: Position of the contact point in the world frame.
            normal: Contact normal in the world frame, pointing from the
                environment to the robot.
            friction: Static friction coefficient.
            max_force: Maximum norm of the contact force, infinite for no
                bound.
        """
        super().__init__(
            position, rotation_from_normal(normal), friction, max_force
        )

    @property
    def wrench_map(self) -> np.ndarray:
        """Map from contact coordinates to the contact wrench."""
        return np.vstack([np.eye(3), np.zeros((3, 3))])

    @property
    def wrench_inequalities(self) -> tuple[np.ndarray, np.ndarray]:
        """Inequalities on contact coordinates."""
        return force_inequalities(self.friction, self.max_force)


class FlatContact(Contact):
    """Flat contact over a rectangular area.

    A flat contact constrains all six degrees of freedom of the contact
    frame, for instance a foot sole on the ground. Contact coordinates are
    the six components of the contact wrench at the center of the rectangle,
    in the contact frame. They lie in the wrench cone of the rectangle, see
    `Stability of surface contacts for humanoid robots
    <https://arxiv.org/abs/1501.04719>`_ (Caron, Pham and Nakamura, ICRA
    2015).

    Attributes:
        half_length: Half-length of the rectangle along the x-axis of the
            contact frame.
        half_width: Half-width of the rectangle along the y-axis of the
            contact frame.
    """

    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        half_length: float,
        half_width: float,
        friction: float,
        max_force: float = np.inf,
    ):
        """Initialize flat contact.

        Args:
            position: Position of the center of the rectangle in the world
                frame.
            rotation: Rotation matrix from the contact frame to the world
                frame. The z-axis of the contact frame is the contact
                normal, pointing from the environment to the robot.
            half_length: Half-length of the rectangle along the x-axis of
                the contact frame.
            half_width: Half-width of the rectangle along the y-axis of the
                contact frame.
            friction: Static friction coefficient.
            max_force: Maximum norm of the resultant contact force, infinite
                for no bound.
        """
        super().__init__(position, rotation, friction, max_force)
        self.half_length = half_length
        self.half_width = half_width

    @property
    def wrench_map(self) -> np.ndarray:
        """Map from contact coordinates to the contact wrench."""
        return np.eye(6)

    @property
    def wrench_inequalities(self) -> tuple[np.ndarray, np.ndarray]:
        """Inequalities on contact coordinates."""
        X, Y = self.half_length, self.half_width
        mu = self.friction / np.sqrt(2.0)  # inner approximation
        # fmt: off
        A = np.array([
            # fx  fy             fz taux  tauy  tauz
            [-1,   0,           -mu,   0,    0,    0],
            [+1,   0,           -mu,   0,    0,    0],
            [0,   -1,           -mu,   0,    0,    0],
            [0,   +1,           -mu,   0,    0,    0],
            [0,    0,            -Y,  -1,    0,    0],
            [0,    0,            -Y,  +1,    0,    0],
            [0,    0,            -X,   0,   -1,    0],
            [0,    0,            -X,   0,   +1,    0],
            [-Y,  -X, -(X + Y) * mu, +mu,  +mu,   -1],
            [-Y,  +X, -(X + Y) * mu, +mu,  -mu,   -1],
            [+Y,  -X, -(X + Y) * mu, -mu,  +mu,   -1],
            [+Y,  +X, -(X + Y) * mu, -mu,  -mu,   -1],
            [+Y,  +X, -(X + Y) * mu, +mu,  +mu,   +1],
            [+Y,  -X, -(X + Y) * mu, +mu,  -mu,   +1],
            [-Y,  +X, -(X + Y) * mu, -mu,  +mu,   +1],
            [-Y,  -X, -(X + Y) * mu, -mu,  -mu,   +1],
        ])
        # fmt: on
        b = np.zeros(A.shape[0])
        if np.isfinite(self.max_force):
            A_force, b_force = force_inequalities(np.inf, self.max_force)
            A = np.vstack([A, np.hstack([A_force, np.zeros((5, 3))])])
            b = np.hstack([b, b_force])
        return A, b
