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
import pyroki.robots_config as robots_config


def verify_com_in_polygon(robot,
                          joint_config,
                          foot_link_indices,
                          robot_description,
                          base_pose=None):
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

    # Define local foot corners using configuration
    local_corners = robots_config.compute_foot_local_corners(robot_description=robot_description)

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


def test_com_support_polygon_basic(visualize=False):
    """Test basic COM support polygon cost with G1 humanoid robot."""
    # Define robot description
    robot_description = "g1_description"

    # Load G1 humanoid robot
    urdf_string = load_robot_description(robot_description)
    robot = pk.Robot.from_urdf(urdf_string)

    # Get foot links from robot configuration
    foot_link_names = robots_config.get_foot_link_names(robot_description)
    foot_link_indices = jnp.array([robot.links.names.index(name) for name in foot_link_names])

    # Create joint variable
    joint_var = robot.joint_var_cls(0)

    # Test parameters
    num_directions = 8

    # Create cost and use it in a minimal optimization problem
    factors = [
        pk.costs.com_support_polygon_cost(
            robot,
            joint_var,
            foot_link_indices,
            robot_description,
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

    # Visualize if requested
    if visualize:
        import viser
        from viser.extras import ViserUrdf

        # Create viser server
        server = viser.ViserServer()
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Add robot visualization
        base_frame = server.scene.add_frame("/base", show_axes=False)
        urdf_vis = ViserUrdf(server, urdf_string, root_node_name="/base")
        urdf_vis.update_cfg(np.array(optimized_config))

        # Create support polygon visualizer
        support_viz = pk.viewer.SupportPolygonVisualizer(
            server,
            robot,
            robot_description,
            root_node_name="/support_polygon_basic",
            com_color=(255, 50, 50),
            polygon_color=(50, 255, 50),
        )

        # Update visualization
        support_viz.update(optimized_config)

        # Add status information
        status_text = server.gui.add_text("Status", support_viz.get_status_text(), disabled=True)

        close_button = server.gui.add_button("Close Visualization")

        print(f"\nVisualization server running at: http://localhost:{server.get_port()}")
        print("Click 'Close Visualization' button or Ctrl+C to continue...")

        # Wait for close button or interrupt
        try:
            while True:
                if close_button.value:
                    break
                # Update status text
                status_text.value = support_viz.get_status_text()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        support_viz.remove()
        server.stop()

    # Verify results using the visualizer's internal calculations
    test_viz = pk.viewer.SupportPolygonVisualizer(
        None,    # No server needed for calculation only
        robot,
        robot_description,
        visible=False)
    test_viz.update(optimized_config)

    assert test_viz.is_inside, f"COM at {test_viz.com_xy} is outside support polygon!"

    print(f"✓ Basic COM support polygon cost test passed!")
    print(f"  - COM position (X,Y): ({test_viz.com_xy[0]:.3f}, {test_viz.com_xy[1]:.3f})")
    print(f"  - Minimum margin to boundary: {test_viz.min_margin:.3f} m")

    return optimized_config


def test_com_support_polygon_with_base(visualize=False):
    """Test COM support polygon cost with mobile base for G1 humanoid."""
    # Define robot description
    robot_description = "g1_description"

    # Load G1 humanoid robot
    urdf_string = load_robot_description(robot_description)
    robot = pk.Robot.from_urdf(urdf_string)

    # Get foot links from robot configuration
    foot_link_names = robots_config.get_foot_link_names(robot_description)
    foot_link_indices = jnp.array([robot.links.names.index(name) for name in foot_link_names])

    # Create variables
    joint_var = robot.joint_var_cls(0)
    base_var = jaxls.SE3Var(0)

    # Test parameters
    num_directions = 16

    # Create cost and use it in a minimal optimization problem
    factors = [
        pk.costs.com_support_polygon_cost_with_base(
            robot,
            joint_var,
            base_var,
            foot_link_indices,
            robot_description,
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

    # Visualize if requested
    if visualize:
        import viser
        from viser.extras import ViserUrdf

        # Create viser server
        server = viser.ViserServer()
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Add base frame and robot visualization
        base_frame = server.scene.add_frame("/base", show_axes=False)
        urdf_vis = ViserUrdf(server, urdf_string, root_node_name="/base")

        # Update robot configuration
        base_frame.position = np.array(optimized_base.translation())
        wxyz_xyz = optimized_base.wxyz_xyz
        base_frame.wxyz = np.array(wxyz_xyz[:4])    # First 4 elements are wxyz quaternion
        urdf_vis.update_cfg(np.array(optimized_config))

        # Create support polygon visualizer
        support_viz = pk.viewer.SupportPolygonVisualizer(
            server,
            robot,
            robot_description,
            root_node_name="/support_polygon_mobile",
            com_color=(255, 100, 100),
            polygon_color=(100, 255, 100),
            show_com_line=True,
            com_line_height=0.8,
        )

        # Update visualization
        support_viz.update(optimized_config, optimized_base)

        # Add status information
        status_text = server.gui.add_text("Status", support_viz.get_status_text(), disabled=True)

        base_text = server.gui.add_text(
            "Base Position", f"({optimized_base.translation()[0]:.3f}, "
            f"{optimized_base.translation()[1]:.3f}, {optimized_base.translation()[2]:.3f})",
            disabled=True)

        close_button = server.gui.add_button("Close Visualization")

        print(f"\nVisualization server running at: http://localhost:{server.get_port()}")
        print("Click 'Close Visualization' button or Ctrl+C to continue...")

        # Wait for close button or interrupt
        try:
            while True:
                if close_button.value:
                    break
                # Update status text
                status_text.value = support_viz.get_status_text()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        support_viz.remove()
        server.stop()

    # Verify results using the visualizer's internal calculations
    test_viz = pk.viewer.SupportPolygonVisualizer(
        None,    # No server needed for calculation only
        robot,
        robot_description,
        visible=False)
    test_viz.update(optimized_config, optimized_base)

    assert test_viz.is_inside, f"COM at {test_viz.com_xy} is outside support polygon!"

    print(f"✓ COM support polygon cost with base test passed!")
    print(f"  - COM position (X,Y): ({test_viz.com_xy[0]:.3f}, {test_viz.com_xy[1]:.3f})")
    print(f"  - Minimum margin to boundary: {test_viz.min_margin:.3f} m")
    print(f"  - Base position: ({optimized_base.translation()[0]:.3f}, "
          f"{optimized_base.translation()[1]:.3f}, {optimized_base.translation()[2]:.3f})")

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
