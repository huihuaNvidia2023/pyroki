"""Mobile Humanoid IK

Solve humanoid IK with multiple targets and a mobile base.
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def main():
    """Main function for humanoid IK with mobile base."""

    urdf = load_robot_description("g1_description")
    target_link_names = [
        "left_ankle_roll_link", "right_ankle_roll_link", "left_palm_link", "right_palm_link"
    ]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    print(f"Number of joints: {robot.joints.num_joints}")
    print(f"Number of actuated joints: {robot.joints.num_actuated_joints}")
    print(f"Joint names: {robot.joints.names}")
    print(f"Actuated joint names: {robot.joints.actuated_names}")

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    torso_height = 0.0
    ik_target_left_ankle = server.scene.add_transform_controls("/ik_target_left_ankle",
                                                               scale=0.2,
                                                               position=(0.05, 0.1,
                                                                         torso_height - 0.7),
                                                               wxyz=(1, 0, 0, 0))
    ik_target_right_ankle = server.scene.add_transform_controls("/ik_target_right_ankle",
                                                                scale=0.2,
                                                                position=(0.05, -0.1,
                                                                          torso_height - 0.7),
                                                                wxyz=(1, 0, 0, 0))
    ik_target_left_palm = server.scene.add_transform_controls("/ik_target_left_palm",
                                                              scale=0.2,
                                                              position=(0.41, 0.2,
                                                                        torso_height + 0.1),
                                                              wxyz=(1, 0, 0, 0))
    ik_target_right_palm = server.scene.add_transform_controls("/ik_target_right_palm",
                                                               scale=0.2,
                                                               position=(0.41, -0.2,
                                                                         torso_height + 0.1),
                                                               wxyz=(1, 0, 0, 0))
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Add GUI controls for base constraints
    with server.gui.add_folder("Base Constraints"):
        fix_x = server.gui.add_checkbox("Fix X", False)
        fix_y = server.gui.add_checkbox("Fix Y", False)
        fix_z = server.gui.add_checkbox("Fix Z", True)    # Usually want to keep humanoid height
        fix_roll = server.gui.add_checkbox("Fix Roll", True)    # Usually want humanoid upright
        fix_pitch = server.gui.add_checkbox("Fix Pitch", True)    # Usually want humanoid upright
        fix_yaw = server.gui.add_checkbox("Fix Yaw", False)    # Allow rotation

    # Initialize configuration
    cfg = np.array(robot.joint_var_cls(0).default_factory())
    base_pos = np.array([0.0, 0.0, torso_height])
    base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    while True:
        # Solve IK with mobile base.
        start_time = time.time()
        base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([
                ik_target_left_ankle.position, ik_target_right_ankle.position,
                ik_target_left_palm.position, ik_target_right_palm.position
            ]),
            target_wxyzs=np.array([
                ik_target_left_ankle.wxyz, ik_target_right_ankle.wxyz, ik_target_left_palm.wxyz,
                ik_target_right_palm.wxyz
            ]),
            fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
            fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
            prev_pos=base_pos,
            prev_wxyz=base_wxyz,
            prev_cfg=cfg,
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = base_pos
        base_frame.wxyz = base_wxyz


if __name__ == "__main__":
    main()
