# SPDX-License-Identifier: GPL-3.0-or-later

"""Inverse kinematics of Microban with Mink.

Each end effector and the pelvis follow a mocap target. In the viewer, double-click a target
to select it, then drag it with Ctrl + right click (translation) or Ctrl + left
click (rotation).
"""

import mink
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from microban_sim import build_spec

from climbing_sim_utils import (
    FEET,
    HANDS,
    add_sky,
    compute_polygon,
    draw_polygon_and_com,
    standing_configuration,
)

# Computing the polygon takes 10 to 15 ms, more than one IK step
POLYGON_PERIOD = 10  # [steps]

# Keys that enable or disable limbs: digits 6 to 9 are free in MuJoCo's viewer
TOGGLE_KEYS = {
    ord("6"): "left_hand_ball",
    ord("7"): "right_hand_ball",
    ord("8"): "left_foot_ball",
    ord("9"): "right_foot_ball",
}

INSTRUCTIONS = """
Move the translucent green spheres, targets of the hands, feet and pelvis:

- Select a target: double-click it.
- Move it in the plane facing the camera: Ctrl + right drag.
- Move it in the horizontal plane: Ctrl + Shift + right drag.
- Rotate it: Ctrl + left drag.

The pelvis target is inside the trunk, at hip height. If a target is hidden
inside the robot, press 2 to hide meshes, select the target, then press 2.

Press 6 and 7 to disable or enable the left and right hands, 8 and 9 for the
left and right feet. A disabled limb has a grey target, no task, and no
contact in the polygon.
"""

TARGET_RGBA = [0.1, 0.53, 0.33, 0.4]
DISABLED_TARGET_RGBA = [0.5, 0.5, 0.5, 0.4]


def load_model() -> mujoco.MjModel:
    """Load the climbing robot on its floor, with one mocap target per task."""
    spec = build_spec("climber")
    add_sky(spec)
    for frame in [*FEET, *HANDS]:
        # Spheres slightly larger than the balls (radius 12.5 mm) they track
        target = spec.worldbody.add_body(name=f"{frame}_target", mocap=True)
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.016, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            rgba=TARGET_RGBA,
        )
    # Point target of the pelvis, inside the trunk
    target = spec.worldbody.add_body(name="pelvis_target", mocap=True)
    target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.016, 0.0, 0.0],
        contype=0,
        conaffinity=0,
        rgba=TARGET_RGBA,
    )
    return spec.compile()


if __name__ == "__main__":
    model = load_model()
    configuration = mink.Configuration(model)

    # The trunk is the floating base, with its frame at hip height. Only the
    # position of the pelvis is a target, its orientation is left free.
    pelvis_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    feet_tasks = [
        mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        for foot in FEET
    ]
    # Arms have three joints, enough for the position of their ball tips
    hand_tasks = [
        mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for hand in HANDS
    ]
    limb_tasks = dict(zip([*FEET, *HANDS], [*feet_tasks, *hand_tasks]))
    limits = [mink.ConfigurationLimit(model)]

    model = configuration.model
    data = configuration.data
    configuration.update(standing_configuration(model))

    mink.move_mocap_to_frame(model, data, "pelvis_target", "trunk", "body")
    for frame in [*FEET, *HANDS]:
        mink.move_mocap_to_frame(model, data, f"{frame}_target", frame, "site")

    com = data.subtree_com[model.body("trunk").id]  # view, updated by IK
    polygon = compute_polygon(model, data)

    # Disabled limbs have no task and no contact in the polygon
    enabled = {site: True for site in limb_tasks}
    polygon_sites = list(enabled)

    def key_callback(key: int) -> None:
        """Enable or disable a limb when its key is pressed."""
        if (site := TOGGLE_KEYS.get(key)) is None:
            return
        enabled[site] = not enabled[site]
        geom_id = model.body(f"{site}_target").geomadr[0]
        rgba = TARGET_RGBA if enabled[site] else DISABLED_TARGET_RGBA
        model.geom_rgba[geom_id] = rgba

    print(INSTRUCTIONS, flush=True)
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.lookat[:] = com
        viewer.cam.distance = 1.0  # [m]
        viewer.cam.azimuth = 120.0  # [deg]
        viewer.cam.elevation = -15.0  # [deg]
        rate = RateLimiter(frequency=200.0, warn=False)
        step = 0
        while viewer.is_running():
            active_sites = [site for site, on in enabled.items() if on]
            pelvis_task.set_target(
                mink.SE3.from_mocap_name(model, data, "pelvis_target")
            )
            for site in active_sites:
                limb_tasks[site].set_target(
                    mink.SE3.from_mocap_name(model, data, f"{site}_target")
                )
            tasks = [pelvis_task, *(limb_tasks[site] for site in active_sites)]

            velocity = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                solver="daqp",
                damping=1e-3,
                limits=limits,
            )
            configuration.integrate_inplace(velocity, rate.dt)
            mujoco.mj_camlight(model, data)

            step += 1
            if step % POLYGON_PERIOD == 0 or active_sites != polygon_sites:
                polygon = compute_polygon(model, data, active_sites)
                polygon_sites = active_sites

            viewer.user_scn.ngeom = 0
            draw_polygon_and_com(viewer.user_scn, polygon, com)

            viewer.sync()
            rate.sleep()
