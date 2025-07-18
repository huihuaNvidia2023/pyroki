"""Mobile Humanoid Trajectory Optimization with Data Logging

This example demonstrates how to log entire trajectories using the data logging module.
"""

import time
import numpy as np
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


def main():
    """Main function for mobile humanoid trajectory optimization with logging."""

    # Load robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create data logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/humanoid_trajopt",
        contact_threshold=0.01,
        record_velocities=True,
        record_accelerations=True,
    )

    # Define link names
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    hand_link_names = ["left_palm_link", "right_palm_link"]

    # Fixed foot positions (standing pose)
    torso_height = 0.75
    foot_positions = np.array([
        [0.05, 0.1, 0.05],    # Left foot (matching ankle z-position from 13)
        [0.05, -0.1, 0.05],    # Right foot (matching ankle z-position from 13)
    ])
    foot_wxyzs = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)    # Both feet flat on ground

    # Hand start and end positions
    hand_start_positions = np.array([
        [0.41, 0.2, torso_height + 0.1],    # Left hand
        [0.41, -0.2, torso_height + 0.1],    # Right hand
    ])
    hand_end_positions = np.array([
        [0.41, 0.2, torso_height + 0.3],    # Left hand reaches forward and up
        [0.41, -0.2, torso_height + 0.3],    # Right hand reaches forward and up
    ])
    hand_wxyzs = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)    # Palms facing down

    # Trajectory parameters
    timesteps = 30
    dt = 0.05    # 50ms per step

    # Base constraints - allow movement in xy and yaw, fix z, roll, pitch
    fix_base_position = (False, False, False)    # Free in x,y; fixed in z
    fix_base_orientation = (False, False, False)    # Fixed roll,pitch; free yaw

    # Solve trajectory optimization
    print("Solving trajectory optimization...")
    initial_base_pos = np.array([0.0, 0.0, torso_height])  # Start at torso height
    initial_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity orientation
    com_support_weight = 500.0
    com_support_margin = 0.0
    
    base_positions, base_wxyzs, joint_cfgs = pks.solve_trajopt_with_base(
        robot=robot,
        foot_link_names=foot_link_names,
        hand_link_names=hand_link_names,
        foot_positions=foot_positions,
        foot_wxyzs=foot_wxyzs,
        hand_start_positions=hand_start_positions,
        hand_start_wxyzs=hand_wxyzs,
        hand_end_positions=hand_end_positions,
        hand_end_wxyzs=hand_wxyzs,
        fix_base_position=fix_base_position,
        fix_base_orientation=fix_base_orientation,
        timesteps=timesteps,
        dt=dt,
        prev_pos=initial_base_pos,
        prev_wxyz=initial_base_wxyz,
        com_support_weight=com_support_weight,
        com_support_margin=com_support_margin,
    )
    print("Trajectory optimization complete!")

    # Visualize
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)

    # Add base frame for robot
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

    # Visualize start and end hand positions
    for i, (name, color) in enumerate(zip(["left", "right"], [(255, 0, 0), (0, 0, 255)])):
        # Start position
        server.scene.add_frame(
            f"/hand_{name}_start",
            position=hand_start_positions[i],
            wxyz=hand_wxyzs[i],
            axes_length=0.05,
            axes_radius=0.01,
        )
        server.scene.add_label(
            f"/hand_{name}_start/label",
            text=f"{name} start",
            position=(0, 0, 0.05),
        )

        # End position
        server.scene.add_frame(
            f"/hand_{name}_end",
            position=hand_end_positions[i],
            wxyz=hand_wxyzs[i],
            axes_length=0.05,
            axes_radius=0.01,
        )
        server.scene.add_label(
            f"/hand_{name}_end/label",
            text=f"{name} end",
            position=(0, 0, 0.05),
        )

    # Visualize foot positions
    for i, name in enumerate(["left", "right"]):
        server.scene.add_frame(
            f"/foot_{name}",
            position=foot_positions[i],
            wxyz=foot_wxyzs[i],
            axes_length=0.03,
            axes_radius=0.005,
        )

    # Playback controls
    timestep_slider = server.gui.add_slider("Timestep",
                                            min=0,
                                            max=timesteps - 1,
                                            step=1,
                                            initial_value=0)
    playing = server.gui.add_checkbox("Playing", initial_value=True)
    speed_slider = server.gui.add_slider("Playback Speed",
                                         min=0.1,
                                         max=2.0,
                                         step=0.1,
                                         initial_value=1.0)

    # Add trajectory info
    with server.gui.add_folder("Trajectory Info"):
        duration_text = server.gui.add_text("Duration", f"{timesteps * dt:.1f}s", disabled=True)
        base_x_text = server.gui.add_number("Base X", 0.0, disabled=True)
        base_y_text = server.gui.add_number("Base Y", 0.0, disabled=True)
        base_yaw_text = server.gui.add_number("Base Yaw", 0.0, disabled=True)

    # Add visualization controls
    with server.gui.add_folder("Visualization"):
        show_support_polygon = server.gui.add_checkbox("Show Support Polygon", True)
        show_support_polygon.on_update(
            lambda _: support_viz.set_visibility(show_support_polygon.value))
        com_status_text = server.gui.add_text("COM Status",
                                              support_viz.get_status_text(),
                                              disabled=True)

    # Add logging controls
    with server.gui.add_folder("Data Logging"):
        record_traj_button = server.gui.add_button("Record Trajectory")
        record_traj_button.on_click(lambda _: record_trajectory())
        
        recording_status = server.gui.add_text("Status", "Ready to record", disabled=True)
        saved_path = server.gui.add_text("Saved to", "", disabled=True)

    def record_trajectory():
        """Record the entire trajectory to a parquet file."""
        recording_status.value = "Recording trajectory..."
        
        # Start new episode with trajectory metadata
        metadata = {
            'trajectory_type': 'mobile_humanoid_reach',
            'timesteps': timesteps,
            'dt': dt,
            'total_duration': timesteps * dt,
            'com_support_weight': com_support_weight,
            'com_support_margin': com_support_margin,
            'hand_start_positions': hand_start_positions.tolist(),
            'hand_end_positions': hand_end_positions.tolist(),
            'foot_positions': foot_positions.tolist(),
        }
        logger.start_episode(custom_metadata=metadata)
        
        # Record each timestep
        for t in range(timesteps):
            # Create base pose
            base_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.array(base_wxyzs[t])),
                jnp.array(base_positions[t])
            )
            
            # Record frame with timestamp
            logger.record_frame(
                joint_cfg=joint_cfgs[t],
                base_pose=base_pose,
                timestamp=t * dt
            )
        
        # Save episode
        filepath = logger.save_episode()
        recording_status.value = "Trajectory recorded!"
        saved_path.value = str(filepath)
        
        print(f"Trajectory saved to: {filepath}")

    # Main visualization loop
    last_time = time.time()
    
    while True:
        current_time = time.time()

        if playing.value:
            # Update timestep based on real time and playback speed
            time_delta = (current_time - last_time) * speed_slider.value
            timestep_delta = int(time_delta / dt)

            if timestep_delta > 0:
                timestep_slider.value = (timestep_slider.value + timestep_delta) % timesteps
                last_time = current_time

        # Update robot visualization
        t = timestep_slider.value
        base_frame.position = base_positions[t]
        base_frame.wxyz = base_wxyzs[t]
        urdf_vis.update_cfg(joint_cfgs[t])

        # Update info display
        base_x_text.value = float(base_positions[t][0])
        base_y_text.value = float(base_positions[t][1])

        # Extract yaw from quaternion using JAX
        w, x, y, z = base_wxyzs[t]
        base_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        base_yaw_text.value = float(np.degrees(base_yaw))

        # Update support polygon visualization
        # Convert base pose to SE3 for the visualizer
        base_pose_SE3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(
                jnp.array([base_wxyzs[t][1], base_wxyzs[t][2], base_wxyzs[t][3], base_wxyzs[t][0]])),
            jnp.array(base_positions[t]))
        support_viz.update(jnp.array(joint_cfgs[t]), base_pose_SE3)

        # Update COM status text
        com_status_text.value = support_viz.get_status_text()

        time.sleep(0.01)    # Small sleep to prevent CPU spinning


if __name__ == "__main__":
    main() 