# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared code of the examples built on the hackathon simulator, climbing-sim."""

from collections.abc import Iterable
from itertools import pairwise

import mujoco
import numpy as np
import pypoman
from climbing_sim import JOINTS, ClimbingSim, Wall
from climbing_sim.sim import STAND
from scipy.spatial import QhullError

from static_balance import (
    ContactSet,
    PointContact,
    compute_static_equilibrium_polygon,
    compute_supporting_forces,
)

# Centers of the balls at the ends of the climbing hands and feet
FEET = ["right_foot_ball", "left_foot_ball"]
HANDS = ["right_hand_ball", "left_hand_ball"]

# Balls rest in cups that open upward. The cups are frictionless, but their
# surface normals span a cone of half-angle 80 degrees around the vertical,
# which we model as a friction cone with the corresponding coefficient.
FRICTION = np.tan(np.radians(80.0))

# Ballpark of the force an XL330 can hold at a ball: its peak torque is
# kt * max_current = 0.346 N.m/A * 1.75 A = 0.61 N.m (BAM model m6 in the
# simulator), which in the hanging posture lets a ball carry 4 to 8 N
# depending on the direction of the force.
MAX_FORCE = 5.0  # [N]

GREEN = np.array([0.1, 0.53, 0.33, 1.0])
RED = np.array([0.86, 0.21, 0.27, 1.0])


def add_sky(spec: mujoco.MjSpec) -> None:
    """Add a gradient skybox to a scene."""
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )


def standing_configuration(model: mujoco.MjModel) -> np.ndarray:
    """Configuration of the robot standing in front of the wall."""
    q = model.qpos0.copy()
    trunk = model.jnt_qposadr[model.body("trunk").jntadr[0]]
    q[trunk : trunk + 3] = STAND["trunk_pos"]
    q[trunk + 3 : trunk + 7] = STAND["trunk_quat"]
    for joint, angle in STAND["joints"].items():
        q[model.joint(joint).qposadr[0]] = angle
    return q


def hanging_pose(wall: Wall) -> dict:
    """Pose of the simulator's robot hanging on the start holds of a wall.

    Args:
        wall: Wall with start holds.

    Returns:
        Pose in the format of ``climbing_sim.sim.STAND``: trunk position,
        trunk orientation as a (w, x, y, z) quaternion, and joint angles.
    """
    sim = ClimbingSim(wall, cameras=[])
    sim.reset_hanging(settle_s=0.0)
    trunk_pos, trunk_quat = sim.trunk_pose()
    joints = {
        joint: float(sim.data.qpos[sim.model.joint(joint).qposadr[0]])
        for joint in JOINTS
    }
    return {"trunk_pos": trunk_pos, "trunk_quat": trunk_quat, "joints": joints}


def ball_contacts(positions: Iterable[np.ndarray], max_force: float) -> ContactSet:
    """Point contacts of balls in their cups.

    Args:
        positions: Positions of the balls in contact, in the world frame.
        max_force: Maximum force of each contact.

    Returns:
        Contact set with one point contact per ball.
    """
    return ContactSet(
        PointContact(position, [0.0, 0.0, 1.0], FRICTION, max_force)
        for position in positions
    )


def compute_polygon_at(
    positions: Iterable[np.ndarray],
    mass: float,
    max_force: float = MAX_FORCE,
) -> np.ndarray | None:
    """Static-equilibrium polygon of point contacts at given ball positions.

    Args:
        positions: Positions of the balls in contact, in the world frame.
        mass: Mass of the robot.
        max_force: Maximum force of each contact.

    Returns:
        Vertices of the polygon, or None if there is no static equilibrium.
    """
    positions = list(positions)
    if not positions or max_force <= 0.0:  # no equilibrium without forces
        return None
    try:
        return compute_static_equilibrium_polygon(
            ball_contacts(positions, max_force), mass
        )
    except (pypoman.Exception, QhullError):  # infeasible or degenerate
        return None


def compute_forces_at(
    positions: Iterable[np.ndarray],
    com: np.ndarray,
    mass: float,
) -> list[np.ndarray] | None:
    """Contact forces of minimum norm that hold the robot in equilibrium.

    Contacts have friction cones but no maximum force, so that forces can go
    above the limits of the actuators.

    Args:
        positions: Positions of the balls in contact, in the world frame.
        com: Position of the center of mass, in the world frame.
        mass: Mass of the robot.

    Returns:
        Force of each contact, or None if there is no static equilibrium.
    """
    positions = list(positions)
    if not positions:  # no equilibrium without contacts
        return None
    return compute_supporting_forces(ball_contacts(positions, np.inf), com, mass)


def compute_polygon(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sites: Iterable[str] = (*FEET, *HANDS),
) -> np.ndarray | None:
    """Static-equilibrium polygon of point contacts at some of the balls."""
    mass = model.body_subtreemass[model.body("trunk").id]
    return compute_polygon_at((data.site(site).xpos for site in sites), mass)


def add_geom(scene: mujoco.MjvScene, rgba: np.ndarray) -> mujoco.MjvGeom | None:
    """Initialize a new geometry in a scene, if there is room left."""
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return geom


def draw_polygon(
    scene: mujoco.MjvScene,
    vertices: np.ndarray,
    height: float,
    rgba: np.ndarray,
    width: float = 0.0008,
    fill_alpha: float = 0.3,
) -> None:
    """Draw a horizontal convex polygon: transparent surface, thin edges."""
    points = np.hstack([vertices, np.full((len(vertices), 1), height)])
    fill_rgba = np.array([*rgba[:3], fill_alpha])
    # MuJoCo draws a triangle geom with vertices at pos, pos + mat[:, 0] and
    # pos + mat[:, 1], so we fill the polygon with a fan of triangles
    for second, third in pairwise(points[1:]):
        if (geom := add_geom(scene, fill_rgba)) is None:
            return
        geom.type = mujoco.mjtGeom.mjGEOM_TRIANGLE
        geom.size[:] = 1.0
        geom.pos[:] = points[0]
        mat = np.column_stack([second - points[0], third - points[0], [0, 0, 1]])
        geom.mat[:] = mat
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        if (geom := add_geom(scene, rgba)) is None:
            return
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)


def draw_sphere(
    scene: mujoco.MjvScene, position: np.ndarray, radius: float, rgba: np.ndarray
) -> None:
    """Draw a sphere."""
    if (geom := add_geom(scene, rgba)) is not None:
        geom.size[:] = radius
        geom.pos[:] = position


def is_inside(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Check whether a 2D point lies in a convex polygon with CCW vertices."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    offsets = point - vertices
    return bool(np.all(edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0] >= 0))


def draw_polygon_and_com(
    scene: mujoco.MjvScene, polygon: np.ndarray | None, com: np.ndarray
) -> None:
    """Draw the polygon at the height of the CoM, green if it contains it."""
    if polygon is not None and is_inside(com[:2], polygon):
        color = GREEN
    else:
        color = RED
    if polygon is not None:
        draw_polygon(scene, polygon, com[2], color)
    draw_sphere(scene, com, 0.008, color)
