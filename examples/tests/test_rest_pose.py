#!/usr/bin/env python
"""Test custom rest pose functionality for IK and trajectory optimization."""

import sys
import os
# Add parent directory to path to import pyroki_snippets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyroki as pk
from pyroki_snippets import solve_ik_with_multiple_targets_and_base
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main():
    # Load G1 humanoid robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)
    
    # Get foot link names
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    
    # Define simple test targets
    left_foot_pos = np.array([-0.1, 0.1, 0.0])
    right_foot_pos = np.array([0.1, 0.1, 0.0])
    foot_positions = np.array([left_foot_pos, right_foot_pos])
    foot_wxyzs = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])  # Identity quaternions
    
    # Initial base pose
    prev_pos = np.array([0.0, 0.0, 0.8])
    prev_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    prev_cfg = np.zeros(robot.joints.num_actuated_joints)
    
    # Fix base position/orientation (free base)
    fix_base_position = (False, False, False)
    fix_base_orientation = (False, False, False)
    
    print("Testing IK with default rest pose...")
    # Test 1: Default behavior (no rest pose specified)
    base_pos1, base_wxyz1, cfg1 = solve_ik_with_multiple_targets_and_base(
        robot=robot,
        target_link_names=foot_link_names,
        target_wxyzs=foot_wxyzs,
        target_positions=foot_positions,
        fix_base_position=fix_base_position,
        fix_base_orientation=fix_base_orientation,
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
        prev_cfg=prev_cfg,
    )
    
    print(f"Result with default rest pose:")
    print(f"  Base position: {base_pos1}")
    print(f"  Joint config mean: {cfg1.mean():.4f}")
    print(f"  Joint config range: [{cfg1.min():.4f}, {cfg1.max():.4f}]")
    
    print("\nTesting IK with custom rest pose (all zeros)...")
    # Test 2: Custom rest pose (all zeros)
    rest_joint_pose = np.zeros(robot.joints.num_actuated_joints)
    rest_base_pose = (np.array([0.0, 0.0, 0.9]), np.array([1.0, 0.0, 0.0, 0.0]))
    
    base_pos2, base_wxyz2, cfg2 = solve_ik_with_multiple_targets_and_base(
        robot=robot,
        target_link_names=foot_link_names,
        target_wxyzs=foot_wxyzs,
        target_positions=foot_positions,
        fix_base_position=fix_base_position,
        fix_base_orientation=fix_base_orientation,
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
        prev_cfg=prev_cfg,
        rest_joint_pose=rest_joint_pose,
        rest_base_pose=rest_base_pose,
    )
    
    print(f"Result with custom rest pose (zeros):")
    print(f"  Base position: {base_pos2}")
    print(f"  Joint config mean: {cfg2.mean():.4f}")
    print(f"  Joint config range: [{cfg2.min():.4f}, {cfg2.max():.4f}]")
    
    # Compare results
    print(f"\nDifference in base position: {np.linalg.norm(base_pos1 - base_pos2):.6f}")
    print(f"Difference in joint config: {np.linalg.norm(cfg1 - cfg2):.6f}")
    
    # Test 3: Try trajectory optimization
    print("\nTesting trajectory optimization...")
    from pyroki_snippets import solve_trajopt_with_base
    
    hand_link_names = ["left_palm_link", "right_palm_link"]
    hand_start_positions = np.array([[0.3, 0.3, 0.6], [0.3, -0.3, 0.6]])
    hand_end_positions = np.array([[0.4, 0.3, 0.7], [0.4, -0.3, 0.7]])
    hand_wxyzs = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    
    # Without custom rest pose
    base_positions1, base_wxyzs1, joint_cfgs1 = solve_trajopt_with_base(
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
        timesteps=10,
        dt=0.1,
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
    )
    
    # With custom rest pose
    base_positions2, base_wxyzs2, joint_cfgs2 = solve_trajopt_with_base(
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
        timesteps=10,
        dt=0.1,
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
        rest_joint_pose=rest_joint_pose,
        rest_base_pose=rest_base_pose,
    )
    
    print(f"Trajectory optimization results:")
    print(f"  Mean joint config (default): {joint_cfgs1.mean():.4f}")
    print(f"  Mean joint config (custom): {joint_cfgs2.mean():.4f}")
    print(f"  Difference: {np.abs(joint_cfgs1.mean() - joint_cfgs2.mean()):.6f}")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 