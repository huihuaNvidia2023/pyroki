"""Cost function for COM support polygon margin."""

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from .._robot import Robot
from ..robots_config import compute_foot_local_corners


def com_support_polygon_cost(
    robot: Robot,
    joint_var: Var[Array],
    foot_link_indices: Array,
    robot_description: str,
    num_directions: int,
    weight: Array | float,
    margin_threshold: float = 0.0,
) -> Cost:
    """Creates a cost for COM support polygon margin.
    
    Assumes Z-axis points upward and projects COM onto XY plane (ground plane).
    
    This factory function precomputes the sampling directions to avoid JAX tracing issues.
    """

    # Precompute directions here, before JAX tracing
    angles = jnp.linspace(0, 2 * jnp.pi, num_directions, endpoint=False)
    precomputed_directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)    # [D, 2]

    # Precompute foot corners here too, capturing robot_description in closure
    local_corners = compute_foot_local_corners(robot_description)

    @Cost.create_factory
    def _com_support_polygon_cost_impl(
        vals: VarValues,
        robot: Robot,
        joint_var: Var[Array],
        foot_link_indices: Array,
        weight: Array | float,
        margin_threshold: float,
    ) -> Array:
        """Computes residual penalizing COM outside the support polygon."""
        # Get joint configuration
        cfg = vals[joint_var]

        # Get link poses via forward kinematics
        link_poses = robot.forward_kinematics(cfg)    # Shape: [num_links, 7]

        # Extract foot link poses
        foot_poses_wxyz_xyz = link_poses[foot_link_indices]    # Shape: [F, 7]
        foot_poses = jaxlie.SE3(foot_poses_wxyz_xyz)

        # Apply transformation for each foot using precomputed local corners
        def transform_foot_corners(foot_pose):
            """Transform local corners to world coordinates for one foot."""
            world_corners = jax.vmap(foot_pose.apply)(local_corners)
            return world_corners

        # Map over all feet
        world_corners = jax.vmap(transform_foot_corners)(foot_poses)    # [F, 4, 3]

        # Extract XY coordinates (ignore Z/height)
        corners_xy = world_corners[:, :, [0, 1]]    # [F, 4, 2]
        corners_xy_flat = corners_xy.reshape(-1, 2)    # [F*4, 2]

        # Compute COM in world frame
        all_link_positions = jaxlie.SE3(link_poses).translation()    # [num_links, 3]

        # Get link masses
        masses = robot.links.masses[:, None]    # [num_links, 1]
        total_mass = masses.sum()

        # Handle case where all masses are zero (fallback to uniform distribution)
        com_position = jnp.where(total_mass > 1e-6,
                                 (all_link_positions * masses).sum(axis=0) / total_mass,
                                 all_link_positions.mean(axis=0))

        # Project COM to XY plane
        com_xy = jnp.array([com_position[0], com_position[1]])    # [2]

        # Use precomputed directions from closure
        directions = precomputed_directions

        # Project polygon corners and COM onto each direction
        corners_proj = corners_xy_flat @ directions.T    # [F*4, D]
        com_proj = com_xy @ directions.T    # [D]

        # Compute support function (max projection) for each direction
        hull_support = corners_proj.max(axis=0)    # [D]

        # Compute signed margins (positive if COM inside polygon)
        margins = hull_support - com_proj    # [D]

        # Create residuals: penalize when margin < threshold
        residuals = jnp.maximum(0.0, margin_threshold - margins)

        return (residuals * weight).flatten()

    # Return the cost created without passing robot_description as an argument
    return _com_support_polygon_cost_impl(
        robot,
        joint_var,
        foot_link_indices,
        weight,
        margin_threshold,
    )


def com_support_polygon_cost_with_base(
    robot: Robot,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    foot_link_indices: Array,
    robot_description: str,
    num_directions: int,
    weight: Array | float,
    margin_threshold: float = 0.0,
) -> Cost:
    """Creates a cost for COM support polygon margin with mobile base.
    
    Assumes Z-axis points upward and projects COM onto XY plane (ground plane).
    
    This factory function precomputes the sampling directions to avoid JAX tracing issues.
    """

    # Precompute directions here, before JAX tracing
    angles = jnp.linspace(0, 2 * jnp.pi, num_directions, endpoint=False)
    precomputed_directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)    # [D, 2]

    # Precompute foot corners here too, capturing robot_description in closure
    local_corners = compute_foot_local_corners(robot_description=robot_description)

    @Cost.create_factory
    def _com_support_polygon_cost_with_base_impl(
        vals: VarValues,
        robot: Robot,
        joint_var: Var[Array],
        T_world_base_var: Var[jaxlie.SE3],
        foot_link_indices: Array,
        weight: Array | float,
        margin_threshold: float,
    ) -> Array:
        """COM support polygon cost for robots with mobile base."""
        # Get joint configuration and base transform
        cfg = vals[joint_var]
        T_world_base = vals[T_world_base_var]

        # Get link poses in base frame
        link_poses_base = robot.forward_kinematics(cfg)    # Shape: [num_links, 7]

        # Transform each link pose to world frame
        def transform_pose_to_world(pose_wxyz_xyz):
            pose_base = jaxlie.SE3(pose_wxyz_xyz)
            pose_world = T_world_base @ pose_base
            return pose_world.parameters()

        link_poses_world = jax.vmap(transform_pose_to_world)(
            link_poses_base)    # Shape: [num_links, 7]

        # Extract foot link poses
        foot_poses_wxyz_xyz = link_poses_world[foot_link_indices]    # Shape: [F, 7]
        foot_poses = jaxlie.SE3(foot_poses_wxyz_xyz)

        # Transform corners to world using precomputed local corners
        def transform_foot_corners(foot_pose):
            world_corners = jax.vmap(foot_pose.apply)(local_corners)
            return world_corners

        world_corners = jax.vmap(transform_foot_corners)(foot_poses)    # [F, 4, 3]
        corners_xy = world_corners[:, :, [0, 1]].reshape(-1, 2)    # [F*4, 2]

        # Compute COM in world frame
        link_poses_world_SE3 = jaxlie.SE3(link_poses_world)
        all_link_positions_world = link_poses_world_SE3.translation()

        # Get link masses
        masses = robot.links.masses[:, None]
        total_mass = masses.sum()

        # Handle case where all masses are zero (fallback to uniform distribution)
        com_position = jnp.where(total_mass > 1e-6,
                                 (all_link_positions_world * masses).sum(axis=0) / total_mass,
                                 all_link_positions_world.mean(axis=0))

        com_xy = jnp.array([com_position[0], com_position[1]])

        # Use precomputed directions from closure
        directions = precomputed_directions

        # Compute margins
        corners_proj = corners_xy @ directions.T
        com_proj = com_xy @ directions.T
        hull_support = corners_proj.max(axis=0)
        margins = hull_support - com_proj

        # Residuals
        residuals = jnp.maximum(0.0, margin_threshold - margins)
        return (residuals * weight).flatten()

    # Return the cost created without passing robot_description as an argument
    return _com_support_polygon_cost_with_base_impl(
        robot,
        joint_var,
        T_world_base_var,
        foot_link_indices,
        weight,
        margin_threshold,
    )
