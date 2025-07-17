"""Simple demo of the PyRoKi data logging module.

This script demonstrates basic usage of the data logging functionality
without requiring a GUI or complex setup.
"""

import numpy as np
import jax.numpy as jnp
import jaxlie
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description


def demo_single_frame_logging():
    """Demonstrate logging a single frame."""
    print("\n=== Single Frame Logging Demo ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/demo",
        contact_threshold=0.05,
    )

    # Start episode with metadata
    metadata = {
        'demo_type': 'single_frame',
        'robot_config': 'home_position',
    }
    logger.start_episode(custom_metadata=metadata)

    # Record a single frame at home position
    joint_cfg = np.zeros(robot.joints.num_actuated_joints)
    logger.record_frame(joint_cfg)

    # Save episode
    filepath = logger.save_episode()
    print(f"Saved single frame to: {filepath}")

    # Load and display info
    loaded = pk.logging.DataLogger.load_episode(filepath)
    print(f"Loaded episode with {len(loaded)} frames")
    print(f"Robot: {loaded.metadata.robot_name}")
    print(f"Custom metadata: {loaded.metadata.custom_metadata}")


def demo_trajectory_logging():
    """Demonstrate logging a trajectory."""
    print("\n=== Trajectory Logging Demo ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create logger with velocity recording
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/demo",
        contact_threshold=0.05,
        record_velocities=True,
        record_accelerations=True,
    )

    # Start episode
    metadata = {
        'demo_type': 'trajectory',
        'motion_type': 'sinusoidal',
        'duration': 2.0,
    }
    logger.start_episode(custom_metadata=metadata)

    # Generate and record sinusoidal trajectory
    timesteps = 20
    dt = 0.1
    for t in range(timesteps):
        # Simple sinusoidal motion
        joint_cfg = 0.5 * np.sin(2 * np.pi * t * dt / 2.0) * np.ones(
            robot.joints.num_actuated_joints)
        logger.record_frame(joint_cfg, timestamp=t * dt)

    # Save episode
    filepath = logger.save_episode()
    print(f"Saved trajectory to: {filepath}")

    # Load and display info
    loaded = pk.logging.DataLogger.load_episode(filepath)
    print(f"Loaded trajectory with {len(loaded)} frames")
    print(f"Duration: {loaded.metadata.duration:.2f} seconds")
    print(f"First frame joint positions: {loaded.frames[0].joint_positions[:3]}...")
    print(f"Last frame joint positions: {loaded.frames[-1].joint_positions[:3]}...")


def demo_mobile_base_logging():
    """Demonstrate logging with a mobile base."""
    print("\n=== Mobile Base Logging Demo ===")

    # Load humanoid robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create logger
    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/demo",
        contact_threshold=0.01,    # Lower for humanoid
    )

    # Start episode
    metadata = {
        'demo_type': 'mobile_base',
        'motion_type': 'circular_path',
    }
    logger.start_episode(custom_metadata=metadata)

    # Generate circular base motion
    timesteps = 20
    dt = 0.1
    radius = 0.5

    for t in range(timesteps):
        # Circular motion for base
        angle = 2 * np.pi * t / timesteps
        base_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.75])
        base_wxyz = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])    # Rotate with motion

        # Keep joints at default
        joint_cfg = np.array(robot.joint_var_cls(0).default_factory())

        # Create SE3 pose
        base_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(base_wxyz)),
                                                             jnp.array(base_pos))

        # Record frame
        logger.record_frame(joint_cfg, base_pose=base_pose, timestamp=t * dt)

    # Save episode
    filepath = logger.save_episode()
    print(f"Saved mobile base trajectory to: {filepath}")

    # Load and analyze
    loaded = pk.logging.DataLogger.load_episode(filepath)
    print(f"Loaded trajectory with {len(loaded)} frames")

    # Check contact detection
    contacts_per_frame = [np.sum(frame.contact_status) for frame in loaded.frames]
    print(f"Average contacts per frame: {np.mean(contacts_per_frame):.1f}")

    # Show base motion
    print(f"Base start position: {loaded.frames[0].root_position}")
    print(f"Base end position: {loaded.frames[-1].root_position}")


def demo_custom_contact_detector():
    """Demonstrate custom contact detection."""
    print("\n=== Custom Contact Detector Demo ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Custom contact detector that marks end effector as always in contact
    def custom_contact_detector(robot, link_positions, link_names, **kwargs):
        contacts = np.zeros(len(link_names), dtype=bool)
        # Mark specific links as in contact
        for i, name in enumerate(link_names):
            if 'hand' in name or 'finger' in name:
                contacts[i] = True
        return contacts

    # Create logger with custom compute functions
    compute_fns = pk.logging.ComputeFunctions(detect_contact=custom_contact_detector)

    logger = pk.logging.DataLogger(
        robot=robot,
        output_dir="logs/demo",
        compute_functions=compute_fns,
    )

    # Record a frame
    logger.start_episode(custom_metadata={'demo_type': 'custom_contact'})
    logger.record_frame(np.zeros(robot.joints.num_actuated_joints))
    filepath = logger.save_episode()

    # Load and check contacts
    loaded = pk.logging.DataLogger.load_episode(filepath)
    frame = loaded.frames[0]

    print("Links detected as in contact:")
    for i, (name, contact) in enumerate(zip(frame.link_names, frame.contact_status)):
        if contact:
            print(f"  - {name}")


if __name__ == "__main__":
    print("PyRoKi Data Logging Module Demo")
    print("================================")

    # Run all demos
    demo_single_frame_logging()
    demo_trajectory_logging()
    demo_mobile_base_logging()
    demo_custom_contact_detector()

    print("\n✓ All demos completed successfully!")
    print("Check the 'logs/demo' directory for the generated parquet files.")
