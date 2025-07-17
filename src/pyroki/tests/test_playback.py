"""Test script to verify logging and playback integration.

This script runs a simple logging demo and then plays back the logged data.
"""

import pyroki as pk
import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description
import tempfile
from pathlib import Path


def test_logging_and_playback():
    """Test logging some data and playing it back."""

    print("=== Testing Logging and Playback ===")

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")

        # 1. Create and log some robot data
        print("\n1. Creating robot and logging data...")
        urdf = load_robot_description("panda_description")
        robot = pk.Robot.from_urdf(urdf)

        # Create logger
        logger = pk.logging.DataLogger(
            robot=robot,
            output_dir=temp_dir,
            contact_threshold=0.05,
            record_velocities=True,
        )

        # Log a simple trajectory
        logger.start_episode(custom_metadata={
            'test': 'playback_integration',
            'motion': 'sinusoidal',
        })

        timesteps = 20
        dt = 0.1
        for t in range(timesteps):
            joint_cfg = 0.3 * np.sin(2 * np.pi * t * dt / 2.0) * np.ones(
                robot.joints.num_actuated_joints)
            logger.record_frame(joint_cfg, timestamp=t * dt)

        filepath = logger.save_episode()
        print(f"Saved episode to: {filepath}")

        # 2. Play back the logged data
        print("\n2. Playing back logged data...")
        print("Check your browser for the visualization")
        print("Controls:")
        print("  - Click 'Play' to animate through the trajectory")
        print("  - Use the Frame slider to scrub through frames")
        print("  - Press Ctrl+C to exit")

        # Create playback viewer
        player = pk.viewer.EpisodePlayback(
            episodes=temp_dir,    # Will find all episodes in the directory
            show_support_polygon=False)

        # Run the viewer
        try:
            player.run()
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_logging_and_playback()
