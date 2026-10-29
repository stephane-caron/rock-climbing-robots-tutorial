# SPDX-License-Identifier: GPL-3.0-or-later

"""Inverse kinematics of Microban with Pinker, visualized in Viser.

Each end effector and the pelvis follow a target to drag around in the Viser
tab of your Web browser. In the control panel, checkboxes select the contacts
of the static-equilibrium polygon, a slider sets the maximum force of each
contact, and a live plot shows the L1 norm of each contact force in the
distribution of minimum norm.
"""

import microban_model
import numpy as np
import pinker
import viser
import viser.transforms as vtf
import viser.uplot
from climbing_sim import Wall
from loop_rate_limiters import RateLimiter
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask
from pinker.visualizer import start_viser_visualizer
from static_balance.static_equilibrium import GRAVITY

from climbing_sim_utils import (
    FEET,
    GREEN,
    HANDS,
    MAX_FORCE,
    RED,
    compute_forces_at,
    compute_polygon_at,
    hanging_pose,
    is_inside,
)

# Competition wall of the hackathon, from level N1 to N3
WALL = "N2"

# An IK step takes about 0.6 ms, but displaying the robot takes 9 ms and
# computing the polygon 10 to 15 ms, so we do them less often
DISPLAY_PERIOD = 4  # [steps]
POLYGON_PERIOD = 10  # [steps]

# Number of samples in the live plot of contact forces (3 s at 50 Hz)
PLOT_SAMPLES = 150

PLOT_COLORS = {
    "left_hand_ball": "#1f77b4",
    "right_hand_ball": "#ff7f0e",
    "left_foot_ball": "#9467bd",
    "right_foot_ball": "#8c564b",
}

INSTRUCTIONS = """
In the browser tab of the visualizer:

- Drag target handles to move the hands, feet and pelvis: arrows translate,
  rings rotate.
- Contacts: checkboxes select which contacts count in the polygon (all tasks
  stay active).
- Max force: maximum force of each contact, in newtons.
- The plot shows the L1 norm of each contact force, the quantity that the
  polygon bounds by the maximum force (dashed line).
"""

LIMB_LABELS = {
    "left_hand_ball": "Left hand",
    "right_hand_ball": "Right hand",
    "left_foot_ball": "Left foot",
    "right_foot_ball": "Right foot",
}


def configuration_from_pose(model: kin.Model, pose: dict) -> np.ndarray:
    """Configuration vector from a pose in the format of the simulator."""
    q = pinker.custom_configuration(model, **pose["joints"])
    w, x, y, z = pose["trunk_quat"]
    q[0:3] = pose["trunk_pos"]
    q[3:7] = [x, y, z, w]  # Pinker quaternions are (x, y, z, w)
    return q


def draw_polygon_and_com(
    server: viser.ViserServer,
    polygon: np.ndarray | None,
    com: np.ndarray,
) -> None:
    """Draw the polygon at the height of the CoM, green if it contains it."""
    inside = polygon is not None and is_inside(com[:2], polygon)
    color = tuple(int(c) for c in 255 * (GREEN if inside else RED)[:3])
    server.scene.add_icosphere("/com", radius=0.008, color=color, position=com)
    if polygon is None:
        server.scene.remove_by_name("/polygon")
        return
    points = np.hstack([polygon, np.full((len(polygon), 1), com[2])])
    faces = np.array([[0, i, i + 1] for i in range(1, len(points) - 1)])
    server.scene.add_mesh_simple(
        "/polygon/fill",
        vertices=points,
        faces=faces,
        color=color,
        opacity=0.3,
        side="double",
    )
    edges = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    server.scene.add_line_segments(
        "/polygon/edges",
        edges,
        colors=color,
        thickness=0.002,  # [m]
    )


if __name__ == "__main__":
    robot = pinker.load_robot_mjcf(microban_model.mjcf(climb=True))
    model = robot.model
    viz = start_viser_visualizer(robot)
    server = viz.viewer

    # Start with the ball of each hand and foot in the socket of its start
    # hold, as in the simulator's hanging start
    q_init = configuration_from_pose(model, hanging_pose(Wall.load(WALL)))
    configuration = pinker.Configuration(model, robot.data, q_init)
    viz.display(configuration.q)

    com = kin.center_of_mass(model, configuration.data, configuration.q)
    mass = configuration.data.mass[0]

    # The trunk is the floating base, with its frame at hip height. Only the
    # position of the pelvis is a target, its orientation is left free.
    pelvis_task = FrameTask(
        "trunk",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=0.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    feet_tasks = [
        FrameTask(
            foot,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1.0,
        )
        for foot in FEET
    ]
    hand_tasks = [
        FrameTask(
            hand,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1.0,
        )
        for hand in HANDS
    ]
    limb_tasks = dict(zip([*FEET, *HANDS], [*feet_tasks, *hand_tasks]))

    # Targets to drag around, rotating only those with an orientation cost
    target_handles = {}
    for task in [pelvis_task, *feet_tasks, *hand_tasks]:
        task.set_target_from_configuration(configuration)
        target = task.transform_target_to_world
        target_handles[task.frame] = server.scene.add_transform_controls(
            f"/targets/{task.frame}",
            position=target.translation,
            wxyz=vtf.SO3.from_matrix(target.rotation).wxyz,
            # The pelvis target is inside the trunk: draw it larger and on top
            scale=0.15 if task is pelvis_task else 0.05,
            depth_test=task is not pelvis_task,
            disable_rotations=task is pelvis_task,
        )
    server.scene.add_grid(
        "/grid",
        width=1.0,
        height=1.0,
        cell_size=0.05,
        position=(com[0], com[1], 0.0),
    )
    camera_position = (com[0] + 0.5, com[1] - 0.8, 0.5)
    server.initial_camera.position = camera_position
    server.initial_camera.look_at = tuple(com)
    for client in server.get_clients().values():
        client.camera.position = camera_position
        client.camera.look_at = tuple(com)

    max_force_slider = server.gui.add_slider(
        "Max force [N]",
        min=0.0,
        max=mass * GRAVITY,
        step=0.1,
        initial_value=MAX_FORCE,
    )
    # All tasks stay active, checkboxes only select contacts of the polygon
    with server.gui.add_folder("Contacts"):
        contact_checkboxes = {
            site: server.gui.add_checkbox(label, initial_value=True)
            for site, label in LIMB_LABELS.items()
        }

    # Live plot of each contact force, We plot the L1 norm |f_x| + |f_y| + f_z.
    forces_text = server.gui.add_markdown("")
    plot_time = np.zeros(PLOT_SAMPLES)
    plot_forces = {site: np.zeros(PLOT_SAMPLES) for site in LIMB_LABELS}
    plot_max_force = np.zeros(PLOT_SAMPLES)
    force_plot = server.gui.add_uplot(
        data=(plot_time, *plot_forces.values(), plot_max_force),
        series=(
            viser.uplot.Series(label="t"),
            *(
                viser.uplot.Series(label=label, stroke=PLOT_COLORS[site], width=2)
                for site, label in LIMB_LABELS.items()
            ),
            viser.uplot.Series(
                label="Max force", stroke="#888888", width=1, dash=[5, 5]
            ),
        ),
        title="Contact force L1 norm [N]",
        scales={
            "x": viser.uplot.Scale(time=False, auto=True),
            "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=True),
        aspect=2.0,
    )

    print(INSTRUCTIONS, flush=True)
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    step = 0
    polygon_inputs = None
    while True:
        for frame, handle in target_handles.items():
            task = pelvis_task if frame == "trunk" else limb_tasks[frame]
            task.set_target(
                kin.SE3(
                    vtf.SO3(np.array(handle.wxyz)).as_matrix(),
                    np.array(handle.position),
                )
            )
        velocity = solve_ik(
            configuration,
            [pelvis_task, *limb_tasks.values()],
            dt,
            solver="daqp",
            damping=1e-3,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)

        step += 1
        contacts = [site for site, box in contact_checkboxes.items() if box.value]
        max_force = max_force_slider.value
        inputs = (contacts, max_force)
        update_polygon = step % POLYGON_PERIOD == 0 or inputs != polygon_inputs
        if step % DISPLAY_PERIOD != 0 and not update_polygon:
            rate.sleep()
            continue

        viz.display(configuration.q)
        positions = [
            configuration.get_transform_frame_to_world(site).translation
            for site in contacts
        ]
        com = kin.center_of_mass(model, configuration.data, configuration.q)
        if update_polygon:
            polygon = compute_polygon_at(positions, mass, max_force)
            draw_polygon_and_com(server, polygon, com)
            polygon_inputs = inputs

        # Forces are zero for disabled contacts, or without equilibrium
        forces = compute_forces_at(positions, com, mass)
        force_norms = dict.fromkeys(LIMB_LABELS, 0.0)
        if forces is not None:
            for site, force in zip(contacts, forces):
                force_norms[site] = float(np.abs(force).sum())  # f_z >= 0
        above = [
            LIMB_LABELS[site] for site, norm in force_norms.items() if norm > max_force
        ]
        if forces is None:
            text = "Supporting forces: **none**"
        elif above:
            text = f"Supporting forces: **above max** at {', '.join(above)}"
        else:
            text = "Supporting forces: **within max**"
        if forces_text.content != text:
            forces_text.content = text
        for buffer in (plot_time, *plot_forces.values(), plot_max_force):
            buffer[:-1] = buffer[1:]
        plot_time[-1] = step * dt
        for site, norm in force_norms.items():
            plot_forces[site][-1] = norm
        plot_max_force[-1] = max_force
        force_plot.data = (plot_time, *plot_forces.values(), plot_max_force)
        rate.sleep()
