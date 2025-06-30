"""Test COM support polygon cost functionality."""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import time
import numpy as np
import argparse


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


def visualize_com_and_polygon(urdf_string,
                              robot,
                              joint_config,
                              base_pose,
                              foot_link_indices,
                              foot_dimensions,
                              com_xy,
                              corners_xy,
                              test_name="COM Support Polygon"):
    """Visualize the robot, COM, and support polygon."""
    import viser
    from viser.extras import ViserUrdf

    # Create viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Add base frame and robot visualization
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf_string, root_node_name="/base")

    # Update robot configuration
    if base_pose is not None:
        base_frame.position = np.array(base_pose.translation())
        # Extract quaternion from SE3 (wxyz format)
        wxyz_xyz = base_pose.wxyz_xyz
        base_frame.wxyz = np.array(wxyz_xyz[:4])    # First 4 elements are wxyz quaternion
    # Convert JAX array to numpy for ViserUrdf
    urdf_vis.update_cfg(np.array(joint_config))

    # Visualize support polygon
    # Add polygon corners as small spheres
    for i, corner in enumerate(corners_xy):
        server.scene.add_icosphere(
            f"/polygon/corner_{i}",
            radius=0.01,
            color=(0, 255, 0),    # Green for polygon corners
            position=(float(corner[0]), float(corner[1]), 0.01)    # Slightly above ground
        )

    # Connect polygon corners with lines (simple convex hull visualization)
    # For humanoid with 2 feet, we have 8 corners (4 per foot)
    # We'll draw rectangles for each foot
    num_corners = len(corners_xy)
    if num_corners == 8:    # 2 feet with 4 corners each
        # Draw rectangles for each foot
        for foot_idx in range(2):
            foot_corners = corners_xy[foot_idx * 4:(foot_idx + 1) * 4]
            # Connect corners in order: 0-1-2-3-0
            for i in range(4):
                start = foot_corners[i]
                end = foot_corners[(i + 1) % 4]

                # Create line by adding many small spheres
                num_points = 10
                for j in range(num_points):
                    t = j / (num_points - 1)
                    point = start * (1 - t) + end * t
                    server.scene.add_icosphere(f"/polygon/foot{foot_idx}_edge_{i}_{j}",
                                               radius=0.003,
                                               color=(0, 200, 0),
                                               position=(float(point[0]), float(point[1]), 0.01))
    else:
        # For other cases, just connect consecutive corners
        for i in range(num_corners):
            start = corners_xy[i]
            end = corners_xy[(i + 1) % num_corners]

            num_points = 10
            for j in range(num_points):
                t = j / (num_points - 1)
                point = start * (1 - t) + end * t
                server.scene.add_icosphere(f"/polygon/edge_{i}_{j}",
                                           radius=0.003,
                                           color=(0, 200, 0),
                                           position=(float(point[0]), float(point[1]), 0.01))

    # Visualize COM projection
    # Add COM point
    server.scene.add_icosphere(
        "/com_projection",
        radius=0.02,
        color=(255, 0, 0),    # Red for COM
        position=(float(com_xy[0]), float(com_xy[1]), 0.02)    # Slightly above ground
    )

    # Add vertical line from COM to ground
    com_3d_height = 0.5    # Approximate COM height for visualization
    num_points = 20
    for i in range(num_points):
        z = com_3d_height * (1 - i / (num_points - 1))
        server.scene.add_icosphere(f"/com_line/{i}",
                                   radius=0.002,
                                   color=(200, 0, 0),
                                   position=(float(com_xy[0]), float(com_xy[1]), z))

    # Add labels
    server.scene.add_label("/com_projection/label", text="COM", position=(0, 0, 0.05))

    # Add title and info
    info_text = server.gui.add_text("Test Info", test_name, disabled=True)
    com_text = server.gui.add_text("COM Position",
                                   f"({com_xy[0]:.3f}, {com_xy[1]:.3f})",
                                   disabled=True)

    # Check if COM is inside polygon
    angles = jnp.linspace(0, 2 * jnp.pi, 16, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    corners_proj = corners_xy @ directions.T
    com_proj = com_xy @ directions.T
    margins = corners_proj.max(axis=0) - com_proj
    min_margin = margins.min()
    is_inside = jnp.all(margins >= -1e-6)

    status_text = server.gui.add_text("Status",
                                      "✓ Inside" if is_inside else "✗ Outside",
                                      disabled=True)
    margin_text = server.gui.add_text("Min Margin", f"{min_margin:.3f} m", disabled=True)

    close_button = server.gui.add_button("Close Visualization")

    print(f"\nVisualization server running at: http://localhost:{server.get_port()}")
    print("Click 'Close Visualization' button or Ctrl+C to continue...")

    # Wait for close button or interrupt
    try:
        while True:
            if close_button.value:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    server.stop()


def test_com_support_polygon_basic(visualize=False):
    """Test basic COM support polygon cost with G1 humanoid robot."""
    # Load G1 humanoid robot
    urdf_string = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf_string)

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

    # Visualize if requested
    if visualize:
        visualize_com_and_polygon(urdf_string,
                                  robot,
                                  optimized_config,
                                  None,
                                  foot_link_indices,
                                  foot_dimensions,
                                  com_xy,
                                  corners_xy,
                                  test_name="Basic COM Support Polygon Test")

    return optimized_config


def test_com_support_polygon_with_base(visualize=False):
    """Test COM support polygon cost with mobile base for G1 humanoid."""
    # Load G1 humanoid robot
    urdf_string = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf_string)

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

    # Visualize if requested
    if visualize:
        visualize_com_and_polygon(urdf_string,
                                  robot,
                                  optimized_config,
                                  optimized_base,
                                  foot_link_indices,
                                  foot_dimensions,
                                  com_xy,
                                  corners_xy,
                                  test_name="COM Support Polygon with Mobile Base Test")

    return optimized_config, optimized_base


def main():
    """Run all tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test COM support polygon cost functionality")
    parser.add_argument("--visualize",
                        "-v",
                        action="store_true",
                        help="Enable visualization of results")
    args = parser.parse_args()

    print("Testing COM support polygon cost functions...\n")

    # Test basic version
    joint_config = test_com_support_polygon_basic(visualize=args.visualize)
    print(f"Basic cost test solved, joint config norm: {jnp.linalg.norm(joint_config):.6f}")

    # Test version with base
    joint_config2, base_pose = test_com_support_polygon_with_base(visualize=args.visualize)
    print(f"With-base cost test solved, joint config norm: {jnp.linalg.norm(joint_config2):.6f}")

    print("\n✓ All COM support polygon cost tests passed!")


if __name__ == "__main__":
    main()
