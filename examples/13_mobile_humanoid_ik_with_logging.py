"""Mobile Humanoid IK with Data Logging

This example demonstrates how to integrate the data logging module
with the mobile humanoid IK example.
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np
import argparse
from pathlib import Path
import yourdfpy

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def main():
    """Main function for humanoid IK with mobile base and data logging."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mobile Humanoid IK with Data Logging")
    parser.add_argument(
        "--robot-name",
        "-r",
        type=str,
        help="Robot name for loading from robot_descriptions (e.g., 'g1', 'panda', 'ur5')",
        default="g1"
    )
    parser.add_argument(
        "--urdf",
        "-u",
        type=str,
        help="Path to URDF file (overrides robot-name if provided)",
        default=""
    )
    args = parser.parse_args()
    
    # TODO: Factor out this part into a function.
    # Load URDF either from file or using robot_descriptions
    if args.urdf:
        # Load from URDF file path
        urdf_path = Path(args.urdf)
        if not urdf_path.is_absolute():
            # Make relative paths relative to the workspace root
            urdf_path = Path.cwd() / urdf_path
        
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        print(f"Loading URDF from file: {urdf_path}")
        
        # Define filename handler for resolving mesh paths
        def filename_handler(fname: str) -> str:
            # Handle mesh paths relative to URDF file location
            base_path = urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)
        
        urdf = yourdfpy.URDF.load(str(urdf_path), filename_handler=filename_handler)
        urdf.robot.name = args.robot_name

        all_target_link_names = [
            "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
            "pelvis"
        ]
    else:
        # Load from robot_descriptions using robot name
        print(f"Loading robot description: {args.robot_name}_description")
        urdf = load_robot_description(f"{args.robot_name}_description")

        all_target_link_names = [
            "left_ankle_roll_link", "right_ankle_roll_link", "left_palm_link", "right_palm_link",
            "pelvis"
        ]
    


    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Create data logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/humanoid_ik",
        contact_threshold=0.01,    # Lower threshold for humanoid feet
        record_velocities=True,
        record_accelerations=True,
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create support polygon visualizer
    support_viz = pk.viewer.SupportPolygonVisualizer(server,
                                                     robot,
                                                     root_node_name="/support_polygon",
                                                     com_color=(255, 50, 50),
                                                     polygon_color=(50, 255, 50),
                                                     com_radius=0.03,
                                                     show_com_line=True,
                                                     com_line_height=1.0,
                                                     visible=True)

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

    # Add base transform control
    ik_target_base = server.scene.add_transform_controls("/ik_target_base",
                                                         scale=0.3,
                                                         position=(0.0, 0.0, torso_height),
                                                         wxyz=(1, 0, 0, 0))

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Store fixed foot positions
    fixed_left_ankle_pos = np.array(ik_target_left_ankle.position)
    fixed_right_ankle_pos = np.array(ik_target_right_ankle.position)
    fixed_ankle_wxyz = np.array([1.0, 0.0, 0.0, 0.0])    # Keep feet oriented straight

    # Add GUI controls
    with server.gui.add_folder("IK Options"):
        include_ankle_targets = server.gui.add_checkbox("Include Ankle Targets", True)
        include_ankle_targets.on_update(lambda _: update_ankle_visibility())
        use_v2_solver = server.gui.add_checkbox("Use V2 Solver (Better foot pinning)", False)
        show_base_control = server.gui.add_checkbox("Show Base Control", True)
        show_base_control.on_update(
            lambda _: setattr(ik_target_base, 'visible', show_base_control.value))

    with server.gui.add_folder("Base Constraints"):
        fix_x = server.gui.add_checkbox("Fix X", False)
        fix_y = server.gui.add_checkbox("Fix Y", False)
        fix_z = server.gui.add_checkbox("Fix Z", False)
        fix_roll = server.gui.add_checkbox("Fix Roll", False)
        fix_pitch = server.gui.add_checkbox("Fix Pitch", False)
        fix_yaw = server.gui.add_checkbox("Fix Yaw", False)

    with server.gui.add_folder("COM Support Polygon"):
        com_support_weight = server.gui.add_slider("Weight",
                                                   min=0.0,
                                                   max=1000.0,
                                                   step=0.1,
                                                   initial_value=0.0)
        com_support_margin = server.gui.add_slider("Margin (m)",
                                                   min=0.0,
                                                   max=0.1,
                                                   step=0.01,
                                                   initial_value=0.00)

    with server.gui.add_folder("Base Pose"):
        base_x_text = server.gui.add_text("X (m)", "0.000", disabled=True)
        base_y_text = server.gui.add_text("Y (m)", "0.000", disabled=True)
        base_z_text = server.gui.add_text("Z (m)", "0.750", disabled=True)
        base_roll_text = server.gui.add_text("Roll (deg)", "0.0", disabled=True)
        base_pitch_text = server.gui.add_text("Pitch (deg)", "0.0", disabled=True)
        base_yaw_text = server.gui.add_text("Yaw (deg)", "0.0", disabled=True)

    with server.gui.add_folder("Visualization"):
        show_support_polygon = server.gui.add_checkbox("Show Support Polygon", True)
        show_support_polygon.on_update(
            lambda _: support_viz.set_visibility(show_support_polygon.value))
        com_status_text = server.gui.add_text("COM Status",
                                              support_viz.get_status_text(),
                                              disabled=True)

    # Add logging controls
    with server.gui.add_folder("Data Logging"):
        record_button = server.gui.add_button("Record Single Frame")
        record_button.on_click(lambda _: record_single_frame())

        episode_info = server.gui.add_text("Episode", "No active episode", disabled=True)
        frames_recorded = server.gui.add_number("Frames Recorded", 0, disabled=True)

        save_episode_button = server.gui.add_button("Save Episode")
        save_episode_button.on_click(lambda _: save_and_reset_episode())

    def update_ankle_visibility():
        """Update visibility of ankle transform controls based on checkbox."""
        ik_target_left_ankle.visible = include_ankle_targets.value
        ik_target_right_ankle.visible = include_ankle_targets.value

    # Initially hide ankle controls
    update_ankle_visibility()

    # Initialize configuration
    cfg = np.array(robot.joint_var_cls(0).default_factory())
    base_pos = np.array([0.0, 0.0, torso_height])    # Rest pose
    base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])    # Rest pose

    # Initialize base pose display with rest pose values
    base_x_text.value = f"{base_pos[0]:.3f}"
    base_y_text.value = f"{base_pos[1]:.3f}"
    base_z_text.value = f"{base_pos[2]:.3f}"
    # Compute euler angles from initial quaternion
    import jaxlie
    import jax.numpy as jnp
    initial_so3 = jaxlie.SO3.from_quaternion_xyzw(
        jnp.array([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]]))
    initial_euler = initial_so3.as_rpy_radians()
    base_roll_text.value = f"{float(np.degrees(initial_euler.roll)):.1f}"
    base_pitch_text.value = f"{float(np.degrees(initial_euler.pitch)):.1f}"
    base_yaw_text.value = f"{float(np.degrees(initial_euler.yaw)):.1f}"

    # Track solved values separately
    solved_cfg = None
    solved_base_pos = None
    solved_base_wxyz = None
    ik_has_run = False

    # Recording state
    recording_active = False
    frames_count = 0

    def record_single_frame():
        """Record the current robot state as a single frame."""
        nonlocal recording_active, frames_count, solved_cfg, solved_base_pos, solved_base_wxyz, ik_has_run

        # Check if IK has run at least once
        if not ik_has_run:
            print("Warning: IK solver hasn't run yet. Please wait a moment and try again.")
            return

        # Start new episode if not active
        if not recording_active:
            # Add custom metadata
            metadata = {
                'solver_type': 'V2' if use_v2_solver.value else 'V1',
                'ankle_targets_included': include_ankle_targets.value,
                'com_support_weight': com_support_weight.value,
                'com_support_margin': com_support_margin.value,
            }
            logger.start_episode(custom_metadata=metadata)
            recording_active = True
            frames_count = 0
            episode_info.value = f"Episode {logger.episode_counter} active"

        # Record current frame with solved values
        logger.record_frame(solved_cfg, base_pose=(solved_base_pos, solved_base_wxyz))
        frames_count += 1
        frames_recorded.value = frames_count

        print(f"Recorded frame {frames_count}")

    def save_and_reset_episode():
        """Save the current episode and reset for a new one."""
        nonlocal recording_active, frames_count

        if recording_active and frames_count > 0:
            filepath = logger.save_episode()
            print(f"Episode saved to: {filepath}")

            recording_active = False
            frames_count = 0
            frames_recorded.value = 0
            episode_info.value = "No active episode"
        else:
            print("No active episode to save")

    while True:
        # Get target base pose from transform control
        target_base_pos = np.array(ik_target_base.position)
        target_base_wxyz = np.array(ik_target_base.wxyz)

        # Solve IK with mobile base
        start_time = time.time()

        if use_v2_solver.value and not include_ankle_targets.value:
            # Use V2 solver for better foot pinning
            # Note: V2 solver doesn't support base targets, only uses prev_pos/prev_wxyz for initialization
            base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base_v2(
                robot=robot,
                foot_link_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                hand_link_names=["left_palm_link", "right_palm_link"],
                foot_positions=np.array([fixed_left_ankle_pos, fixed_right_ankle_pos]),
                foot_wxyzs=np.array([fixed_ankle_wxyz, fixed_ankle_wxyz]),
                hand_positions=np.array(
                    [ik_target_left_palm.position, ik_target_right_palm.position]),
                hand_wxyzs=np.array([ik_target_left_palm.wxyz, ik_target_right_palm.wxyz]),
                fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
                fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
                prev_pos=base_pos,    # Use current base for smooth motion
                prev_wxyz=base_wxyz,    # Use current base for smooth motion
                prev_cfg=cfg,
            )
        else:
            # Use V1 solver (original)
            # Determine which targets to use
            if include_ankle_targets.value:
                target_link_names = all_target_link_names
                target_positions = np.array([
                    ik_target_left_ankle.position,
                    ik_target_right_ankle.position,
                    ik_target_left_palm.position,
                    ik_target_right_palm.position,
                    target_base_pos    # Add base target
                ])
                target_wxyzs = np.array([
                    ik_target_left_ankle.wxyz,
                    ik_target_right_ankle.wxyz,
                    ik_target_left_palm.wxyz,
                    ik_target_right_palm.wxyz,
                    target_base_wxyz    # Add base target
                ])
                # Use normal weights when ankles are interactive, lower for base
                pos_weights = np.array([50.0, 50.0, 50.0, 50.0, 5.0])    # Lower weight for base
                ori_weights = np.array([10.0, 10.0, 10.0, 10.0, 0.5])    # Lower weight for base
            else:
                # Use only hand targets, with fixed foot positions
                target_link_names = all_target_link_names    # Still include all for stability
                target_positions = np.array([
                    fixed_left_ankle_pos,
                    fixed_right_ankle_pos,    # Use fixed positions
                    ik_target_left_palm.position,
                    ik_target_right_palm.position,
                    target_base_pos    # Add base target
                ])
                target_wxyzs = np.array([
                    fixed_ankle_wxyz,
                    fixed_ankle_wxyz,    # Fixed orientations
                    ik_target_left_palm.wxyz,
                    ik_target_right_palm.wxyz,
                    target_base_wxyz    # Add base target
                ])
                # Use very high weights for feet to keep them pinned, lower for base
                pos_weights = np.array([100.0, 100.0, 50.0, 50.0, 5.0])    # Lower weight for base
                ori_weights = np.array([20.0, 20.0, 10.0, 10.0, 0.5])    # Lower weight for base

            base_pos, base_wxyz, cfg = pks.solve_ik_with_multiple_targets_and_base(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=target_positions,
                target_wxyzs=target_wxyzs,
                fix_base_position=(fix_x.value, fix_y.value, fix_z.value),
                fix_base_orientation=(fix_roll.value, fix_pitch.value, fix_yaw.value),
                prev_pos=base_pos,    # Use current base for smooth motion
                prev_wxyz=base_wxyz,    # Use current base for smooth motion
                prev_cfg=cfg,
                pos_weights=pos_weights,
                ori_weights=ori_weights,
                com_support_weight=com_support_weight.value,
                com_support_margin=com_support_margin.value,
            )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = base_pos
        base_frame.wxyz = base_wxyz

        # Update support polygon visualization
        # Convert base pose to SE3 for the visualizer
        import jaxlie
        import jax.numpy as jnp
        base_pose_SE3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(
                jnp.array([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]])),
            jnp.array(base_pos))
        support_viz.update(jnp.array(cfg), base_pose_SE3)

        # Update COM status text
        com_status_text.value = support_viz.get_status_text()

        # Update base pose display
        base_x_text.value = f"{base_pos[0]:.3f}"
        base_y_text.value = f"{base_pos[1]:.3f}"
        base_z_text.value = f"{base_pos[2]:.3f}"

        # Convert quaternion to euler angles (using jaxlie which is already imported)
        so3 = jaxlie.SO3.from_quaternion_xyzw(
            jnp.array([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]]))
        euler = so3.as_rpy_radians()    # Returns (roll, pitch, yaw)
        base_roll_text.value = f"{float(np.degrees(euler.roll)):.1f}"
        base_pitch_text.value = f"{float(np.degrees(euler.pitch)):.1f}"
        base_yaw_text.value = f"{float(np.degrees(euler.yaw)):.1f}"

        # Store solved values for recording
        solved_cfg = cfg
        solved_base_pos = base_pos
        solved_base_wxyz = base_wxyz
        ik_has_run = True


if __name__ == "__main__":
    main()
