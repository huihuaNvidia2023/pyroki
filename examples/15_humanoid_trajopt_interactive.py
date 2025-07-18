"""Interactive Humanoid Trajectory Optimization with Data Logging

This example allows users to interactively set start and end poses for
a humanoid robot's hands while keeping feet planted, then optimizes
a trajectory between them. The optimized trajectories can be logged
to parquet files for later analysis or playback.

Workflow:
1. Move interactive markers to desired start positions
2. Click "Capture Start Pose"
3. Move markers to desired end positions  
4. Click "Capture End Pose"
5. Click "Run Optimization" to compute trajectory
6. View the resulting trajectory playback
7. Click "Record Trajectory" to save the trajectory
8. Click "Reset" to start over
"""

import time
import numpy as np
import jax.numpy as jnp
import jaxlie
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki as pk

# Import managers
from trajectory_controller import TrajectoryController
from trajectory_state import TrajectoryStates


def main():
    """Main function for interactive trajectory optimization with logging."""

    # Load robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Define link names
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    hand_link_names = ["left_palm_link", "right_palm_link"]

    # Create Viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)

    # Create data logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/interactive_trajopt",
        contact_threshold=0.01,
        record_velocities=True,
        record_accelerations=True,
    )

    # Create main controller
    controller = TrajectoryController(robot=robot,
                                      urdf=urdf,
                                      server=server,
                                      foot_link_names=foot_link_names,
                                      hand_link_names=hand_link_names)

    # Track if current trajectory has been recorded
    trajectory_recorded = False

    # Add logging controls
    with server.gui.add_folder("Data Logging"):
        record_traj_button = server.gui.add_button("Record Trajectory")
        recording_status = server.gui.add_text("Status", "No trajectory to record", disabled=True)
        saved_path = server.gui.add_text("Saved to", "", disabled=True)

        def record_trajectory(_=None):
            """Record the optimized trajectory to a parquet file."""
            # This line allows the function to modify the 'trajectory_recorded' variable from the enclosing scope.
            nonlocal trajectory_recorded
            # Check if already recorded
            if trajectory_recorded:
                recording_status.value = "Trajectory already recorded"
                print("This trajectory has already been recorded.")
                return

            # Check if we have a trajectory
            if (controller.state.base_positions is None
                    or controller.state.current_state != TrajectoryStates.PLAYBACK):
                recording_status.value = "No optimized trajectory available"
                return

            recording_status.value = "Recording trajectory..."

            # Get trajectory data from controller
            base_positions = controller.state.base_positions
            base_wxyzs = controller.state.base_wxyzs
            joint_cfgs = controller.state.joint_cfgs
            timesteps = controller.state.timesteps
            dt = controller.state.dt

            # Get optimization parameters from control panel
            com_weight = controller.control_panel.com_weight_slider.value
            com_margin = controller.control_panel.com_margin_slider.value
            fix_position, fix_orientation = controller.control_panel.get_base_constraints()

            # Extract hand positions from captured poses
            hand_start_positions = []
            hand_end_positions = []
            for link_name in hand_link_names:
                if link_name in controller.state.start_positions:
                    hand_start_positions.append(
                        controller.state.start_positions[link_name].tolist())
                if link_name in controller.state.end_positions:
                    hand_end_positions.append(controller.state.end_positions[link_name].tolist())

            # Prepare metadata
            metadata = {
                'trajectory_type': 'interactive_humanoid_reach',
                'timesteps': timesteps,
                'dt': dt,
                'total_duration': timesteps * dt,
                'com_support_weight': com_weight,
                'com_support_margin': com_margin,
                'base_constraints': {
                    'fix_position': fix_position,
                    'fix_orientation': fix_orientation,
                },
                'hand_start_positions': hand_start_positions,
                'hand_end_positions': hand_end_positions,
                'foot_link_names': foot_link_names,
                'hand_link_names': hand_link_names,
            }

            # Start new episode
            logger.start_episode(custom_metadata=metadata)

            # Record each timestep
            for t in range(timesteps):
                # Create base pose
                base_pose = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array(base_wxyzs[t])), jnp.array(base_positions[t]))

                # Record frame with timestamp
                logger.record_frame(joint_cfg=joint_cfgs[t], base_pose=base_pose, timestamp=t * dt)

            # Save episode
            filepath = logger.save_episode()
            recording_status.value = "Trajectory recorded!"
            saved_path.value = str(filepath)

            # Mark as recorded
            trajectory_recorded = True

            print(f"Trajectory saved to: {filepath}")

        record_traj_button.on_click(record_trajectory)

    # Hook into the controller's run button to reset the recorded flag
    original_run_optimization = controller.run_optimization

    def run_optimization_with_reset(_=None):
        """Run optimization and reset the recorded flag."""
        nonlocal trajectory_recorded
        # Reset the recorded flag for new trajectory
        trajectory_recorded = False
        saved_path.value = ""
        # Call the original optimization
        original_run_optimization(_)

    # Replace the callback
    controller.control_panel.run_btn.on_click(run_optimization_with_reset)

    # Hook into the reset button to also reset the recorded flag
    original_reset = controller.reset

    def reset_with_flag_reset(_=None):
        """Reset controller and recording flag."""
        nonlocal trajectory_recorded
        trajectory_recorded = False
        saved_path.value = ""
        recording_status.value = "No trajectory to record"
        original_reset(_)

    controller.control_panel.reset_btn.on_click(reset_with_flag_reset)

    print("\n=== Interactive Trajectory Optimization with Data Logging ===")
    print("1. Move the interactive markers to desired start positions")
    print("2. Click 'Capture Start Pose'")
    print("3. Move markers to desired end positions")
    print("4. Click 'Capture End Pose'")
    print("5. Click 'Run Optimization' to compute the trajectory")
    print("6. Use playback controls to view the result")
    print("7. Click 'Record Trajectory' to save to parquet file")
    print("8. Click 'Reset' to start over\n")

    # Main loop
    last_time = time.time()

    while True:
        current_time = time.time()

        # Update trajectory playback
        last_time = controller.update_playback(current_time, last_time)

        # Update visualization
        controller.update_visualization()

        # Update recording status based on controller state
        if controller.state.base_positions is None:
            recording_status.value = "No trajectory to record"
        elif trajectory_recorded:
            recording_status.value = "Trajectory already recorded"
        elif controller.state.current_state == TrajectoryStates.PLAYBACK:
            recording_status.value = "Ready to record"

        # Small sleep to prevent CPU spinning
        time.sleep(0.01)


if __name__ == "__main__":
    main()
