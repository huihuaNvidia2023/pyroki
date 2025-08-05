"""Trajectory Interpolation with IK and Data Recording

This example demonstrates how to interpolate between initial and final target poses,
solve IK at each frame, and record all frames using the data logger.
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


def interpolate_positions(initial_pos, final_pos, num_steps):
    """
    Linearly interpolate between initial and final positions.
    
    Args:
        initial_pos: Initial position (3D array)
        final_pos: Final position (3D array)
        num_steps: Number of interpolation steps
        
    Returns:
        List of interpolated positions
    """
    positions = []
    for i in range(num_steps + 1):
        t = i / num_steps
        pos = initial_pos + t * (final_pos - initial_pos)
        positions.append(pos)
    return positions


def main():
    """Main function for trajectory interpolation with IK and recording."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trajectory Interpolation with IK Recording")
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
    parser.add_argument(
        "--initial-base-pos",
        type=float,
        nargs=3,
        help="Initial base position (x y z)",
        default=[0.0, 0.0, 0.75]
    )
    parser.add_argument(
        "--final-base-pos",
        type=float,
        nargs=3,
        help="Final base position (x y z)",
        default=[0.0, 0.0, 0.4]
    )
    parser.add_argument(
        "--step-size",
        type=float,
        help="Step size for interpolation",
        default=0.05
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for recorded data",
        default="logs/trajectory_interpolation"
    )
    args = parser.parse_args()
    
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
    
    # Create robot
    robot = pk.Robot.from_urdf(urdf)
    
    # Create data logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir=args.output_dir,
        contact_threshold=0.01,    # Lower threshold for humanoid feet
        record_velocities=True,
        record_accelerations=True,
    )
    
    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    
    # Create support polygon visualizer
    support_viz = pk.viewer.SupportPolygonVisualizer(
        server,
        robot,
        root_node_name="/support_polygon",
        com_color=(255, 50, 50),
        polygon_color=(50, 255, 50),
        com_radius=0.03,
        show_com_line=True,
        com_line_height=1.0,
        visible=True
    )
    
    # Calculate number of steps based on distance and step size
    initial_base_pos = np.array(args.initial_base_pos)
    final_base_pos = np.array(args.final_base_pos)
    distance = np.linalg.norm(final_base_pos - initial_base_pos)
    num_steps = int(distance / args.step_size)
    
    print(f"Interpolating from {initial_base_pos} to {final_base_pos}")
    print(f"Distance: {distance:.3f}, Step size: {args.step_size}, Steps: {num_steps}")
    
    # Generate interpolated base positions
    base_positions = interpolate_positions(initial_base_pos, final_base_pos, num_steps)
    
    # Initialize configuration and base pose
    cfg = np.array(robot.joint_var_cls(0).default_factory())
    base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Fixed orientation for now
    
    # Set initial positions for other target links
    # These will remain fixed during interpolation
    torso_height = initial_base_pos[2]
    fixed_left_ankle_pos = np.array([0.05, 0.1, torso_height - 0.7])
    fixed_right_ankle_pos = np.array([0.05, -0.1, torso_height - 0.7])
    fixed_left_palm_pos = np.array([0.41, 0.2, torso_height + 0.1])
    fixed_right_palm_pos = np.array([0.41, -0.2, torso_height + 0.1])
    fixed_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Add GUI elements
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    progress_handle = server.gui.add_slider("Progress", min=0, max=num_steps, step=1, initial_value=0, disabled=True)
    current_frame_handle = server.gui.add_number("Current Frame", 0, disabled=True)
    
    with server.gui.add_folder("Recording Info"):
        episode_info = server.gui.add_text("Episode", "Not started", disabled=True)
        frames_recorded = server.gui.add_number("Frames Recorded", 0, disabled=True)
        start_button = server.gui.add_button("Start Recording")
        
    recording_started = False
    
    def start_recording():
        nonlocal recording_started
        if not recording_started:
            # Start new episode with metadata
            metadata = {
                'interpolation_type': 'linear',
                'initial_base_pos': initial_base_pos.tolist(),
                'final_base_pos': final_base_pos.tolist(),
                'step_size': args.step_size,
                'num_steps': num_steps,
            }
            logger.start_episode(custom_metadata=metadata)
            recording_started = True
            episode_info.value = f"Episode {logger.episode_counter} active"
            print(f"Started recording episode {logger.episode_counter}")
            
            # Start the interpolation and recording process
            record_interpolated_trajectory()
    
    start_button.on_click(lambda _: start_recording())
    
    def record_interpolated_trajectory():
        """Record the entire interpolated trajectory."""
        nonlocal cfg, base_wxyz
        
        for i, base_pos in enumerate(base_positions):
            # Update progress
            progress_handle.value = i
            current_frame_handle.value = i
            
            # Adjust other target positions based on current base height
            current_height = base_pos[2]
            height_diff = current_height - torso_height
            
            # Update ankle and palm positions to maintain relative positions
            current_left_ankle_pos = fixed_left_ankle_pos
            current_right_ankle_pos = fixed_right_ankle_pos
            current_left_palm_pos = fixed_left_palm_pos + np.array([0, 0, height_diff])
            current_right_palm_pos = fixed_right_palm_pos + np.array([0, 0, height_diff])
            
            # Prepare target positions and orientations
            target_positions = np.array([
                current_left_ankle_pos,
                current_right_ankle_pos,
                current_left_palm_pos,
                current_right_palm_pos,
                base_pos  # pelvis/base position
            ])
            
            target_wxyzs = np.array([
                fixed_wxyz,  # left ankle
                fixed_wxyz,  # right ankle
                fixed_wxyz,  # left palm
                fixed_wxyz,  # right palm
                base_wxyz    # pelvis/base
            ])
            
            # Solve IK
            start_time = time.time()
            
            base_pos_result, base_wxyz_result, cfg = pks.solve_ik_with_multiple_targets_and_base(
                robot=robot,
                target_link_names=all_target_link_names,
                target_positions=target_positions,
                target_wxyzs=target_wxyzs,
                fix_base_position=(False, False, False),  # Allow base to move
                fix_base_orientation=(True, True, True),   # Fix orientation for now
                prev_pos=base_pos,
                prev_wxyz=base_wxyz,
                prev_cfg=cfg,
                pos_weights=np.array([100.0, 100.0, 50.0, 50.0, 50.0]),  # High weight for feet
                ori_weights=np.array([20.0, 20.0, 10.0, 10.0, 10.0]),
                com_support_weight=100.0,  # Enable COM support polygon constraint
                com_support_margin=0.02,
            )
            
            # Update timing
            elapsed_time = time.time() - start_time
            timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
            
            # Update visualizer
            urdf_vis.update_cfg(cfg)
            base_frame.position = base_pos_result
            base_frame.wxyz = base_wxyz_result
            
            # Update support polygon visualization
            import jaxlie
            import jax.numpy as jnp
            base_pose_SE3 = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_quaternion_xyzw(
                    jnp.array([base_wxyz_result[1], base_wxyz_result[2], base_wxyz_result[3], base_wxyz_result[0]])),
                jnp.array(base_pos_result))
            support_viz.update(jnp.array(cfg), base_pose_SE3)
            
            # Record frame
            logger.record_frame(cfg, base_pose=(base_pos_result, base_wxyz_result))
            frames_recorded.value = i + 1
            
            # Small delay for visualization
            time.sleep(0.01)
        
        # Save episode when done
        filepath = logger.save_episode()
        print(f"Recording complete! Episode saved to: {filepath}")
        episode_info.value = f"Episode saved: {filepath.name}"
        
    # Keep the server running
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()