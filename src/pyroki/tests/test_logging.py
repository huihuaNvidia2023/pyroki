"""Unit tests for the PyRoKi logging module."""

import numpy as np
import jax.numpy as jnp
import jaxlie
from pathlib import Path
import tempfile
import shutil
import traceback

import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description


def test_single_frame_logging():
    """Test logging a single frame."""
    print("\n=== Testing single frame logging ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create logger
        logger = pk.logging.DataLogger(
            robot=robot,
            output_dir=temp_dir,
            contact_threshold=0.05,
        )

        # Start episode
        logger.start_episode(custom_metadata={'test': 'single_frame'})

        # Record a single frame
        joint_cfg = np.zeros(robot.joints.num_actuated_joints)
        logger.record_frame(joint_cfg)

        # Save episode
        filepath = logger.save_episode()

        # Verify file exists
        assert filepath.exists(), "Saved file does not exist"

        # Load and verify
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)
        assert len(loaded_episode) == 1, f"Expected 1 frame, got {len(loaded_episode)}"
        assert loaded_episode.metadata.robot_name == robot.name, "Robot name mismatch"
        assert loaded_episode.metadata.custom_metadata[
            'test'] == 'single_frame', "Metadata mismatch"

        # Check frame data
        frame = loaded_episode.frames[0]
        assert len(
            frame.joint_names) == robot.joints.num_actuated_joints, "Joint names count mismatch"
        assert len(frame.link_names) == robot.links.num_links, "Link names count mismatch"
        assert np.allclose(frame.joint_positions, joint_cfg), "Joint positions mismatch"

        print("✓ Single frame logging test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_trajectory_logging():
    """Test logging a trajectory."""
    print("\n=== Testing trajectory logging ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create logger
        logger = pk.logging.DataLogger(
            robot=robot,
            output_dir=temp_dir,
            contact_threshold=0.05,
            record_velocities=True,
            record_accelerations=True,
        )

        # Start episode
        logger.start_episode(custom_metadata={'test': 'trajectory'})

        # Record trajectory
        timesteps = 10
        dt = 0.1
        for t in range(timesteps):
            joint_cfg = np.sin(t * 0.1) * np.ones(robot.joints.num_actuated_joints)
            logger.record_frame(joint_cfg, timestamp=t * dt)

        # Save episode
        filepath = logger.save_episode()

        # Load and verify
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)
        assert len(
            loaded_episode) == timesteps, f"Expected {timesteps} frames, got {len(loaded_episode)}"
        assert abs(loaded_episode.metadata.duration -
                   (timesteps - 1) * dt) < 1e-6, "Duration mismatch"

        # Check timestamps
        for i, frame in enumerate(loaded_episode.frames):
            assert abs(frame.timestamp - i * dt) < 1e-6, f"Timestamp mismatch at frame {i}"

        print("✓ Trajectory logging test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_mobile_base_logging():
    """Test logging with mobile base."""
    print("\n=== Testing mobile base logging ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create logger
        logger = pk.logging.DataLogger(robot=robot, output_dir=temp_dir)

        # Start episode
        logger.start_episode()

        # Record frames with mobile base
        joint_cfg = np.zeros(robot.joints.num_actuated_joints)
        base_positions = [[0, 0, 0], [0.1, 0, 0], [0.2, 0.1, 0]]
        base_wxyz = [1, 0, 0, 0]

        for i, pos in enumerate(base_positions):
            base_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(base_wxyz)),
                                                                 jnp.array(pos))
            logger.record_frame(joint_cfg, base_pose=base_pose, timestamp=i * 0.1)

        # Save and load
        filepath = logger.save_episode()
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)

        # Verify base motion
        for i, frame in enumerate(loaded_episode.frames):
            assert np.allclose(frame.root_position,
                               base_positions[i]), f"Base position mismatch at frame {i}"
            assert np.allclose(frame.root_quaternion,
                               base_wxyz), f"Base orientation mismatch at frame {i}"

        print("✓ Mobile base logging test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_contact_detection():
    """Test contact detection functionality."""
    print("\n=== Testing contact detection ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create logger with high contact threshold
        logger = pk.logging.DataLogger(
            robot=robot,
            output_dir=temp_dir,
            contact_threshold=0.5,    # Higher threshold for testing
        )

        # Start episode
        logger.start_episode()

        # Record frame
        joint_cfg = np.zeros(robot.joints.num_actuated_joints)
        logger.record_frame(joint_cfg)

        # Save and load
        filepath = logger.save_episode()
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)

        # Check contact status
        frame = loaded_episode.frames[0]
        # Some links should be below threshold
        assert np.any(frame.contact_status), "No contacts detected with high threshold"

        print("✓ Contact detection test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_custom_compute_functions():
    """Test custom compute functions."""
    print("\n=== Testing custom compute functions ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Custom velocity function that returns ones
        def custom_joint_velocity(robot, positions, **kwargs):
            return np.ones_like(positions)

        # Create logger with custom functions
        compute_fns = pk.logging.ComputeFunctions(compute_joint_velocity=custom_joint_velocity)
        logger = pk.logging.DataLogger(
            robot=robot,
            output_dir=temp_dir,
            compute_functions=compute_fns,
            record_velocities=True,
        )

        # Record two frames
        logger.start_episode()
        logger.record_frame(np.zeros(robot.joints.num_actuated_joints), timestamp=0.0)
        logger.record_frame(np.zeros(robot.joints.num_actuated_joints), timestamp=0.1)

        # Save and load
        filepath = logger.save_episode()
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)

        # Check custom velocity computation
        # First frame should have zero velocities (no previous frame)
        assert np.allclose(loaded_episode.frames[0].joint_velocities,
                           0.0), "First frame velocities should be zero"
        # Second frame should have ones (from custom function)
        assert np.allclose(loaded_episode.frames[1].joint_velocities,
                           1.0), "Second frame velocities should be ones"

        print("✓ Custom compute functions test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_episode_numbering():
    """Test automatic episode numbering."""
    print("\n=== Testing episode numbering ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        logger = pk.logging.DataLogger(robot=robot, output_dir=temp_dir)

        # Record multiple episodes
        for i in range(3):
            logger.start_episode()
            logger.record_frame(np.zeros(robot.joints.num_actuated_joints))
            filepath = logger.save_episode()

            # Check episode number in filename
            assert f"episode_{i+1:03d}" in str(filepath), f"Episode number {i+1} not in filename"

        print("✓ Episode numbering test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_extensible_metadata():
    """Test extensible metadata functionality."""
    print("\n=== Testing extensible metadata ===")

    # Load robot
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        logger = pk.logging.DataLogger(robot=robot, output_dir=temp_dir)

        # Start episode with custom metadata
        custom_meta = {
            'experiment_id': 'test_123',
            'user': 'test_user',
            'parameters': {
                'gain': 1.0,
                'damping': 0.1
            },
            'notes': 'This is a test episode',
        }
        logger.start_episode(custom_metadata=custom_meta)

        # Record frame
        logger.record_frame(np.zeros(robot.joints.num_actuated_joints))

        # Save and load
        filepath = logger.save_episode()
        loaded_episode = pk.logging.DataLogger.load_episode(filepath)

        # Verify custom metadata
        for key, value in custom_meta.items():
            assert loaded_episode.metadata.custom_metadata[
                key] == value, f"Custom metadata mismatch for {key}"

        print("✓ Extensible metadata test passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all logging tests."""
    print("Running PyRoKi logging module tests...")

    tests = [
        test_single_frame_logging,
        test_trajectory_logging,
        test_mobile_base_logging,
        test_contact_detection,
        test_custom_compute_functions,
        test_episode_numbering,
        test_extensible_metadata,
    ]

    failed_tests = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ {test_func.__name__} failed:")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed_tests.append(test_func.__name__)

    print(f"\n{'='*50}")
    if failed_tests:
        print(f"❌ {len(failed_tests)} tests failed:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
    else:
        print("✅ All tests passed!")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
