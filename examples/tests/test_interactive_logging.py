"""Test script for interactive trajectory optimization with logging."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki as pk
from trajectory_state import TrajectoryState, TrajectoryStates

# Create a simple test
def test_states():
    """Test state enum comparisons."""
    state = TrajectoryState()
    print(f"Initial state: {state.current_state}")
    print(f"Is SETUP_START: {state.current_state == TrajectoryStates.SETUP_START}")
    
    # Simulate setting trajectory data
    state.set_trajectory_data(
        np.zeros((10, 3)),  # base_positions
        np.tile([1, 0, 0, 0], (10, 1)),  # base_wxyzs
        np.zeros((10, 7))  # joint_cfgs
    )
    
    print(f"After set_trajectory_data: {state.current_state}")
    print(f"Is PLAYBACK: {state.current_state == TrajectoryStates.PLAYBACK}")
    print(f"String comparison would fail: {state.current_state == 'PLAYBACK'}")

if __name__ == "__main__":
    test_states() 