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
    all_target_link_names = [
        "left_ankle_roll_link", "right_ankle_roll_link", "left_palm_link", "right_palm_link"
    ]
    hand_target_link_names = ["left_palm_link", "right_palm_link"]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    torso_height = 0.75
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

    # Store fixed foot positions
    fixed_left_ankle_pos = np.array(ik_target_left_ankle.position)
    fixed_right_ankle_pos = np.array(ik_target_right_ankle.position)
    fixed_ankle_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Keep feet oriented straight

    # Add GUI controls
    with server.gui.add_folder("IK Options"):
        include_ankle_targets = server.gui.add_checkbox("Include Ankle Targets", False)
        include_ankle_targets.on_update(lambda _: update_ankle_visibility())
        use_v2_solver = server.gui.add_checkbox("Use V2 Solver (Better foot pinning)", True)
    
    with server.gui.add_folder("Base Constraints"):
        fix_x = server.gui.add_checkbox("Fix X", False)
        fix_y = server.gui.add_checkbox("Fix Y", False)
        fix_z = server.gui.add_checkbox("Fix Z", True)    # Usually want to keep humanoid height
        fix_roll = server.gui.add_checkbox("Fix Roll", True)    # Usually want humanoid upright
        fix_pitch = server.gui.add_checkbox("Fix Pitch", True)    # Usually want humanoid upright
        fix_yaw = server.gui.add_checkbox("Fix Yaw", False)    # Allow rotation

    def update_ankle_visibility():
        """Update visibility of ankle transform controls based on checkbox."""
        ik_target_left_ankle.visible = include_ankle_targets.value
        ik_target_right_ankle.visible = include_ankle_targets.value

    # Initially hide ankle controls
    update_ankle_visibility()

    # Initialize configuration
    cfg = np.array(robot.joint_var_cls(0).default_factory())
    base_pos = np.array([0.0, 0.0, torso_height])
    base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    while True:
        # Solve IK with mobile base
        start_time = time.time()
        
        if use_v2_solver.value and not include_ankle_targets.value:
            # Use V2 solver for better foot pinning
            base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base_v2(
                robot=robot,
                foot_link_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                hand_link_names=["left_palm_link", "right_palm_link"],
                foot_positions=np.array([fixed_left_ankle_pos, fixed_right_ankle_pos]),
                foot_wxyzs=np.array([fixed_ankle_wxyz, fixed_ankle_wxyz]),
                hand_positions=np.array([ik_target_left_palm.position, ik_target_right_palm.position]),
                hand_wxyzs=np.array([ik_target_left_palm.wxyz, ik_target_right_palm.wxyz]),
                fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
                fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
                prev_pos=base_pos,
                prev_wxyz=base_wxyz,
                prev_cfg=cfg,
            )
        else:
            # Use V1 solver (original)
            # Determine which targets to use
            if include_ankle_targets.value:
                target_link_names = all_target_link_names
                target_positions = np.array([
                    ik_target_left_ankle.position, ik_target_right_ankle.position,
                    ik_target_left_palm.position, ik_target_right_palm.position
                ])
                target_wxyzs = np.array([
                    ik_target_left_ankle.wxyz, ik_target_right_ankle.wxyz, 
                    ik_target_left_palm.wxyz, ik_target_right_palm.wxyz
                ])
                # Use normal weights when ankles are interactive
                pos_weights = np.array([50.0, 50.0, 50.0, 50.0])
                ori_weights = np.array([10.0, 10.0, 10.0, 10.0])
            else:
                # Use only hand targets, with fixed foot positions
                target_link_names = all_target_link_names  # Still include all for stability
                target_positions = np.array([
                    fixed_left_ankle_pos, fixed_right_ankle_pos,  # Use fixed positions
                    ik_target_left_palm.position, ik_target_right_palm.position
                ])
                target_wxyzs = np.array([
                    fixed_ankle_wxyz, fixed_ankle_wxyz,  # Fixed orientations
                    ik_target_left_palm.wxyz, ik_target_right_palm.wxyz
                ])
                # Use very high weights for feet to keep them pinned
                pos_weights = np.array([100.0, 100.0, 50.0, 50.0])  # 100x higher for feet
                ori_weights = np.array([20.0, 20.0, 10.0, 10.0])  # 100x higher for feet

            base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=target_positions,
                target_wxyzs=target_wxyzs,
                fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
                fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
                prev_pos=base_pos,
                prev_wxyz=base_wxyz,
                prev_cfg=cfg,
                pos_weights=pos_weights,
                ori_weights=ori_weights,
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
