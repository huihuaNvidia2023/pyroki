"""Test COM support polygon cost functionality."""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description


def verify_com_in_polygon(robot, joint_config, foot_link_indices, foot_dimensions, base_pose=None):
    """Verify that the COM is inside the support polygon.
    
    Returns:
        tuple: (is_inside, com_xy, polygon_vertices_xy)
    """
    # Get link poses
    if base_pose is None:
        link_poses = robot.forward_kinematics(joint_config)
        link_poses_SE3 = jaxlie.SE3(link_poses)
    else:
        # Transform to world frame for mobile base case
        link_poses_base = robot.forward_kinematics(joint_config)
        link_poses_base_SE3 = jaxlie.SE3(link_poses_base)
        link_poses_SE3 = jax.vmap(lambda T: base_pose @ T)(link_poses_base_SE3)

    # Compute COM
    all_link_positions = link_poses_SE3.translation()
    masses = robot.links.masses[:, None]
    total_mass = masses.sum()

    com_position = jnp.where(total_mass > 1e-6,
                             (all_link_positions * masses).sum(axis=0) / total_mass,
                             all_link_positions.mean(axis=0))
    com_xy = jnp.array([com_position[0], com_position[1]])

    # Get foot poses and compute support polygon vertices
    if base_pose is None:
        # Direct indexing for basic case
        foot_poses_wxyz_xyz = link_poses[foot_link_indices]
        foot_poses = jaxlie.SE3(foot_poses_wxyz_xyz)
    else:
        # Extract parameters and index for mobile base case
        link_poses_params = link_poses_SE3.parameters()
        foot_poses_params = link_poses_params[foot_link_indices]
        foot_poses = jaxlie.SE3(foot_poses_params)

    # Define local foot corners
    L, W = foot_dimensions
    local_corners = jnp.array([
        [L / 2, W / 2, 0],
        [L / 2, -W / 2, 0],
        [-L / 2, -W / 2, 0],
        [-L / 2, W / 2, 0],
    ])

    # Transform corners to world
    def transform_foot_corners(foot_pose):
        world_corners = jax.vmap(foot_pose.apply)(local_corners)
        return world_corners

    world_corners = jax.vmap(transform_foot_corners)(foot_poses)    # [F, 4, 3]
    corners_xy = world_corners[:, :, [0, 1]].reshape(-1, 2)    # [F*4, 2]

    # Check if COM is inside convex hull of foot corners
    # Using a simple approach: check if COM can be expressed as convex combination
    # For a more robust check, we could compute the actual convex hull
    # Here we'll check if all signed distances are positive

    # Sample directions and check margins
    num_dirs = 16
    angles = jnp.linspace(0, 2 * jnp.pi, num_dirs, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    corners_proj = corners_xy @ directions.T
    com_proj = com_xy @ directions.T
    hull_support = corners_proj.max(axis=0)
    margins = hull_support - com_proj

    # COM is inside if all margins are positive
    is_inside = jnp.all(margins >= -1e-6)    # Small tolerance for numerical errors

    return is_inside, com_xy, corners_xy


def test_com_support_polygon_basic():
    """Test basic COM support polygon cost with G1 humanoid robot."""
    # Load G1 humanoid robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Use actual foot links for the G1 humanoid
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_link_indices = jnp.array([robot.links.names.index(name) for name in foot_link_names])

    # Create joint variable
    joint_var = robot.joint_var_cls(0)

    # Test parameters - more realistic for humanoid feet
    foot_dimensions = (0.15, 0.08)    # 15cm x 8cm (typical humanoid foot size)
    num_directions = 8

    # Create cost and use it in a minimal optimization problem
    factors = [
        pk.costs.com_support_polygon_cost(
            robot,
            joint_var,
            foot_link_indices,
            foot_dimensions,
            num_directions,
            1.0,    # weight
            0.0,    # margin_threshold
        ),
    ]

    # Create and solve a minimal problem
    test_config = jnp.zeros(robot.joints.num_actuated_joints)
    problem = jaxls.LeastSquaresProblem(factors, [joint_var])
    analyzed = problem.analyze()

    # Solve to verify it works
    solution = analyzed.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(test_config)]),
        verbose=False,
    )

    optimized_config = solution[joint_var]

    # Verify COM is inside support polygon
    is_inside, com_xy, corners_xy = verify_com_in_polygon(robot, optimized_config,
                                                          foot_link_indices, foot_dimensions)

    assert is_inside, f"COM at {com_xy} is outside support polygon!"

    # Compute minimum margin for info
    angles = jnp.linspace(0, 2 * jnp.pi, 16, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    corners_proj = corners_xy @ directions.T
    com_proj = com_xy @ directions.T
    margins = corners_proj.max(axis=0) - com_proj
    min_margin = margins.min()

    print(f"✓ Basic COM support polygon cost test passed!")
    print(f"  - COM position (X,Y): ({com_xy[0]:.3f}, {com_xy[1]:.3f})")
    print(f"  - Minimum margin to boundary: {min_margin:.3f} m")

    return optimized_config


def test_com_support_polygon_with_base():
    """Test COM support polygon cost with mobile base for G1 humanoid."""
    # Load G1 humanoid robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    # Use actual foot links for the G1 humanoid
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_link_indices = jnp.array([robot.links.names.index(name) for name in foot_link_names])

    # Create variables
    joint_var = robot.joint_var_cls(0)
    base_var = jaxls.SE3Var(0)

    # Test parameters - more realistic for humanoid feet
    foot_dimensions = (0.15, 0.08)    # 15cm x 8cm
    num_directions = 16

    # Create cost and use it in a minimal optimization problem
    factors = [
        pk.costs.com_support_polygon_cost_with_base(
            robot,
            joint_var,
            base_var,
            foot_link_indices,
            foot_dimensions,
            num_directions,
            2.0,    # weight
            0.05,    # margin_threshold
        ),
    ]

    # Create and solve a minimal problem
    test_config = jnp.zeros(robot.joints.num_actuated_joints)
    test_base = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_quaternion_xyzw(jnp.array([0.0, 0.0, 0.0, 1.0])),    # Identity rotation
        jnp.array([0.0, 0.0, 0.75]))    # Typical humanoid torso height

    problem = jaxls.LeastSquaresProblem(factors, [joint_var, base_var])
    analyzed = problem.analyze()

    # Solve to verify it works
    solution = analyzed.solve(
        initial_vals=jaxls.VarValues.make(
            [joint_var.with_value(test_config),
             base_var.with_value(test_base)]),
        verbose=False,
    )

    optimized_config = solution[joint_var]
    optimized_base = solution[base_var]

    # Verify COM is inside support polygon
    is_inside, com_xy, corners_xy = verify_com_in_polygon(robot, optimized_config,
                                                          foot_link_indices, foot_dimensions,
                                                          optimized_base)

    assert is_inside, f"COM at {com_xy} is outside support polygon!"

    # Compute minimum margin for info
    angles = jnp.linspace(0, 2 * jnp.pi, 16, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    corners_proj = corners_xy @ directions.T
    com_proj = com_xy @ directions.T
    margins = corners_proj.max(axis=0) - com_proj
    min_margin = margins.min()

    print(f"✓ COM support polygon cost with base test passed!")
    print(f"  - COM position (X,Y): ({com_xy[0]:.3f}, {com_xy[1]:.3f})")
    print(f"  - Minimum margin to boundary: {min_margin:.3f} m")
    print(f"  - Base position: ({optimized_base.translation()[0]:.3f}, "
          f"{optimized_base.translation()[1]:.3f}, {optimized_base.translation()[2]:.3f})")

    return optimized_config, optimized_base


def main():
    """Run all tests."""
    print("Testing COM support polygon cost functions...\n")

    # Test basic version
    joint_config = test_com_support_polygon_basic()
    print(f"Basic cost test solved, joint config norm: {jnp.linalg.norm(joint_config):.6f}")

    # Test version with base
    joint_config2, base_pose = test_com_support_polygon_with_base()
    print(f"With-base cost test solved, joint config norm: {jnp.linalg.norm(joint_config2):.6f}")

    print("\n✓ All COM support polygon cost tests passed!")


if __name__ == "__main__":
    main()
