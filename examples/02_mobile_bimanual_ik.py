"""Mobile Bimanual IK

Bimanual IK with a mobile base! Combines features from 02_bimanual_ik.py and 03_mobile_ik.py.
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def main():
    """Main function for bimanual IK with a mobile base."""

    urdf = load_robot_description("yumi_description")
    target_link_names = ["yumi_link_7_r", "yumi_link_7_l"]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controllers with initial positions.
    ik_target_0 = server.scene.add_transform_controls("/ik_target_0",
                                                      scale=0.2,
                                                      position=(0.41, -0.3, 0.56),
                                                      wxyz=(0, 0, 1, 0))
    ik_target_1 = server.scene.add_transform_controls("/ik_target_1",
                                                      scale=0.2,
                                                      position=(0.41, 0.3, 0.56),
                                                      wxyz=(0, 0, 1, 0))
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Add GUI controls for base constraints
    with server.gui.add_folder("Base Constraints"):
        fix_x = server.gui.add_checkbox("Fix X", False)
        fix_y = server.gui.add_checkbox("Fix Y", False)
        fix_z = server.gui.add_checkbox("Fix Z", True)
        fix_roll = server.gui.add_checkbox("Fix Roll", True)
        fix_pitch = server.gui.add_checkbox("Fix Pitch", True)
        fix_yaw = server.gui.add_checkbox("Fix Yaw", False)

    # Initialize configuration
    cfg = np.array(robot.joint_var_cls(0).default_factory())
    base_pos = np.array([0.0, 0.0, 0.0])
    base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    while True:
        # Solve IK with mobile base.
        start_time = time.time()
        base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([ik_target_0.position, ik_target_1.position]),
            target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
            fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
            prev_pos=base_pos,
            prev_wxyz=base_wxyz,
            prev_cfg=cfg,
            # Use default weights (will be 50.0 for position, 10.0 for orientation)
            pos_weights=None,
            ori_weights=None,
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
