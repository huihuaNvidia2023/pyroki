"""
Trajectory optimization with mobile base support.
"""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import pyroki.robots_config as robots_config
from jax.typing import ArrayLike


def solve_trajopt_with_base(
        robot: pk.Robot,
        foot_link_names: Sequence[str],
        hand_link_names: Sequence[str],
        foot_positions: ArrayLike,    # Shape: (num_feet, 3)
        foot_wxyzs: ArrayLike,    # Shape: (num_feet, 4)
        hand_start_positions: ArrayLike,    # Shape: (num_hands, 3)
        hand_start_wxyzs: ArrayLike,    # Shape: (num_hands, 4)
        hand_end_positions: ArrayLike,    # Shape: (num_hands, 3)
        hand_end_wxyzs: ArrayLike,    # Shape: (num_hands, 4)
        fix_base_position: Tuple[bool, bool, bool],
        fix_base_orientation: Tuple[bool, bool, bool],
        timesteps: int,
        dt: float,
        prev_pos: ArrayLike,    # Initial base position
        prev_wxyz: ArrayLike,    # Initial base orientation
        com_support_weight: float = 0.0,
        com_support_margin: float = 0.0,
        rest_joint_pose: ArrayLike = None,
        rest_base_pose: Tuple[ArrayLike, ArrayLike] = None,    # (position, wxyz)
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Solve trajectory optimization for a mobile robot with multiple end-effectors.
    
    Args:
        robot: PyRoKi robot
        foot_link_names: Names of foot links (kept stationary)
        hand_link_names: Names of hand links (moving from start to end)
        foot_positions: Fixed world positions for feet
        foot_wxyzs: Fixed world orientations for feet
        hand_start/end_positions: Start and end positions for hands
        hand_start/end_wxyzs: Start and end orientations for hands
        fix_base_position: Which base position DOFs to fix (x, y, z)
        fix_base_orientation: Which base orientation DOFs to fix (roll, pitch, yaw)
        timesteps: Number of timesteps in trajectory
        dt: Time step size
        prev_pos: Initial base position for IK solving
        prev_wxyz: Initial base orientation for IK solving
        com_support_weight: Weight for COM support polygon cost
        com_support_margin: Margin for COM support polygon
        rest_joint_pose: Custom rest pose for joints. If None, uses mid-point of joint limits
        rest_base_pose: Custom rest pose for base as (position, wxyz). If None, uses identity
        
    Returns:
        base_positions: Shape (timesteps, 3)
        base_wxyzs: Shape (timesteps, 4)
        joint_cfgs: Shape (timesteps, num_joints)
    """
    if isinstance(hand_start_positions, onp.ndarray):
        np = onp
    elif isinstance(hand_start_positions, jnp.ndarray):
        np = jnp
    else:
        raise ValueError(f"Invalid type for ArrayLike: {type(hand_start_positions)}")

    # Get link indices
    foot_link_indices = [robot.links.names.index(name) for name in foot_link_names]
    hand_link_indices = [robot.links.names.index(name) for name in hand_link_names]

    # Combine all link names for IK solving
    all_link_names = foot_link_names + hand_link_names

    # Convert to JAX arrays
    foot_positions = jnp.array(foot_positions)
    foot_wxyzs = jnp.array(foot_wxyzs)
    hand_start_positions = jnp.array(hand_start_positions)
    hand_start_wxyzs = jnp.array(hand_start_wxyzs)
    hand_end_positions = jnp.array(hand_end_positions)
    hand_end_wxyzs = jnp.array(hand_end_wxyzs)

    # Solve IK for start and end configurations
    start_base_pose, start_cfg, end_base_pose, end_cfg = _solve_start_end_iks(
        robot=robot,
        all_link_names=all_link_names,
        foot_link_indices=jnp.array(foot_link_indices),
        hand_link_indices=jnp.array(hand_link_indices),
        foot_positions=foot_positions,
        foot_wxyzs=foot_wxyzs,
        hand_start_positions=hand_start_positions,
        hand_start_wxyzs=hand_start_wxyzs,
        hand_end_positions=hand_end_positions,
        hand_end_wxyzs=hand_end_wxyzs,
        fix_base=jnp.array(fix_base_position + fix_base_orientation),
        prev_pos=jnp.array(prev_pos),
        prev_wxyz=jnp.array(prev_wxyz),
        com_support_weight=com_support_weight,
        com_support_margin=com_support_margin,
        rest_joint_pose=jnp.array(rest_joint_pose) if rest_joint_pose is not None else None,
        rest_base_pose=rest_base_pose,
    )

    # Initialize trajectories by linear interpolation
    init_joint_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # Initialize base trajectory
    start_pos = start_base_pose.translation()
    end_pos = end_base_pose.translation()
    init_base_positions = jnp.linspace(start_pos, end_pos, timesteps)

    # Interpolate rotations using SLERP
    start_rot = start_base_pose.rotation()
    end_rot = end_base_pose.rotation()
    alphas = jnp.linspace(0, 1, timesteps)

    # Proper SLERP: start_rot @ exp(alpha * log(start_rot^-1 @ end_rot))
    relative_rot = start_rot.inverse() @ end_rot
    relative_log = relative_rot.log()
    init_base_rotations = jax.vmap(lambda a: start_rot @ jaxlie.SO3.exp(a * relative_log))(alphas)

    init_base_poses = jaxlie.SE3.from_rotation_and_translation(init_base_rotations,
                                                               init_base_positions)

    # Optimize trajectory
    base_poses, joint_traj = _optimize_trajectory(
        robot=robot,
        foot_link_indices=jnp.array(foot_link_indices),
        hand_link_indices=jnp.array(hand_link_indices),
        foot_positions=foot_positions,
        foot_wxyzs=foot_wxyzs,
        hand_start_positions=hand_start_positions,
        hand_start_wxyzs=hand_start_wxyzs,
        hand_end_positions=hand_end_positions,
        hand_end_wxyzs=hand_end_wxyzs,
        init_base_poses=init_base_poses,
        init_joint_traj=init_joint_traj,
        fix_base=jnp.array(fix_base_position + fix_base_orientation),
        timesteps=timesteps,
        dt=dt,
        com_support_weight=com_support_weight,
        com_support_margin=com_support_margin,
        rest_joint_pose=jnp.array(rest_joint_pose) if rest_joint_pose is not None else None,
        rest_base_pose=rest_base_pose,
    )

    # Extract results
    base_positions = np.array(base_poses.translation())
    base_wxyzs = np.array(base_poses.rotation().wxyz)
    joint_cfgs = np.array(joint_traj)

    return base_positions, base_wxyzs, joint_cfgs


def _solve_start_end_iks(
    robot: pk.Robot,
    all_link_names: Sequence[str],
    foot_link_indices: jax.Array,
    hand_link_indices: jax.Array,
    foot_positions: jax.Array,
    foot_wxyzs: jax.Array,
    hand_start_positions: jax.Array,
    hand_start_wxyzs: jax.Array,
    hand_end_positions: jax.Array,
    hand_end_wxyzs: jax.Array,
    fix_base: jax.Array,
    prev_pos: jax.Array,
    prev_wxyz: jax.Array,
    com_support_weight: float = 0.0,
    com_support_margin: float = 0.0,
    rest_joint_pose: jax.Array = None,
    rest_base_pose: Tuple[jax.Array, jax.Array] = None,
) -> Tuple[jaxlie.SE3, jax.Array, jaxlie.SE3, jax.Array]:
    """Solve IK for start and end configurations separately."""

    # Use differential weights - much higher for feet to keep them pinned
    num_feet = len(foot_link_indices)
    num_hands = len(hand_link_indices)
    pos_weights = jnp.concatenate([
        jnp.full(num_feet, 100.0),    # High weight for feet
        jnp.full(num_hands, 50.0),    # Normal weight for hands
    ])
    ori_weights = jnp.concatenate([
        jnp.full(num_feet, 20.0),    # High weight for feet
        jnp.full(num_hands, 10.0),    # Normal weight for hands
    ])

    # NOTE: The rest_with_base_cost in solve_ik_with_multiple_targets_and_base uses:
    # - joint_var.default_factory() for joint rest pose (can be customized)
    # - identity for base rest pose (always [0,0,0] position, identity rotation)
    # If you need a different base rest pose, you'd need to modify the cost function
    # or create a custom one that accepts a target base pose parameter.

    # Example of how to use the new rest_with_base_cost_custom function:
    # rest_joint_pose = robot.joint_var_cls(0).default_factory()  # or custom joint config
    # rest_base_pose = jaxlie.SE3.from_rotation_and_translation(
    #     jaxlie.SO3(prev_wxyz),  # Use initial orientation as rest
    #     prev_pos                # Use initial position as rest
    # )
    # pk.costs.rest_with_base_cost_custom(
    #     joint_var,
    #     base_var,
    #     rest_joint_pose,
    #     rest_base_pose,
    #     jnp.array([0.01] * robot.joints.num_actuated_joints + [0.1] * 3 + [0.001] * 3)
    # )

    # Extract base constraints
    fix_base_position = fix_base[:3]
    fix_base_orientation = fix_base[3:]

    # Solve start IK
    start_positions = jnp.concatenate([foot_positions, hand_start_positions], axis=0)
    start_wxyzs = jnp.concatenate([foot_wxyzs, hand_start_wxyzs], axis=0)

    from . import solve_ik_with_multiple_targets_and_base

    start_base_pos, start_base_wxyz, start_cfg = solve_ik_with_multiple_targets_and_base(
        robot=robot,
        target_link_names=all_link_names,
        target_wxyzs=start_wxyzs,    # wxyzs comes first
        target_positions=start_positions,    # positions comes second
        fix_base_position=tuple(fix_base_position),
        fix_base_orientation=tuple(fix_base_orientation),
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
        prev_cfg=robot.joint_var_cls(0).default_factory(),
        pos_weights=pos_weights,
        ori_weights=ori_weights,
        com_support_weight=com_support_weight,
        com_support_margin=com_support_margin,
        rest_joint_pose=rest_joint_pose,
        rest_base_pose=rest_base_pose,
    )

    # Solve end IK using start solution as initial guess
    end_positions = jnp.concatenate([foot_positions, hand_end_positions], axis=0)
    end_wxyzs = jnp.concatenate([foot_wxyzs, hand_end_wxyzs], axis=0)

    end_base_pos, end_base_wxyz, end_cfg = solve_ik_with_multiple_targets_and_base(
        robot=robot,
        target_link_names=all_link_names,
        target_wxyzs=end_wxyzs,    # wxyzs comes first
        target_positions=end_positions,    # positions comes second
        fix_base_position=tuple(fix_base_position),
        fix_base_orientation=tuple(fix_base_orientation),
        prev_pos=start_base_pos,
        prev_wxyz=start_base_wxyz,
        prev_cfg=start_cfg,
        pos_weights=pos_weights,
        ori_weights=ori_weights,
        com_support_weight=com_support_weight,
        com_support_margin=com_support_margin,
        rest_joint_pose=rest_joint_pose,
        rest_base_pose=rest_base_pose,
    )

    # Convert to SE3
    start_base_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(start_base_wxyz),
                                                               start_base_pos)
    end_base_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(end_base_wxyz),
                                                             end_base_pos)

    return start_base_pose, start_cfg, end_base_pose, end_cfg


@jdc.jit
def _optimize_trajectory(
    robot: pk.Robot,
    foot_link_indices: jax.Array,
    hand_link_indices: jax.Array,
    foot_positions: jax.Array,
    foot_wxyzs: jax.Array,
    hand_start_positions: jax.Array,
    hand_start_wxyzs: jax.Array,
    hand_end_positions: jax.Array,
    hand_end_wxyzs: jax.Array,
    init_base_poses: jaxlie.SE3,
    init_joint_traj: jax.Array,
    fix_base: jax.Array,
    timesteps: jdc.Static[int],
    dt: jdc.Static[float],
    com_support_weight: float = 0.0,
    com_support_margin: float = 0.0,
    rest_joint_pose: jax.Array = None,
    rest_base_pose: Tuple[jax.Array, jax.Array] = None,
) -> Tuple[jaxlie.SE3, jax.Array]:
    """Optimize the full trajectory."""

    # Create variables
    joint_vars = robot.joint_var_cls(jnp.arange(timesteps))

    # Use standard SE3Var for trajectory
    base_vars = jaxls.SE3Var(jnp.arange(timesteps))

    # Add batch dimensions
    robot_batched = jax.tree.map(lambda x: x[None], robot)

    factors = []

    # Foot constraints throughout trajectory
    for i in range(len(foot_link_indices)):
        foot_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(foot_wxyzs[i]),
                                                             foot_positions[i])
        factors.append(
            pk.costs.pose_cost_with_base_analytic_jac(
                robot_batched,
                joint_vars,
                base_vars,
                jax.tree.map(lambda x: jnp.repeat(x[None], timesteps, axis=0), foot_pose),
                jnp.full(timesteps, foot_link_indices[i]),
                1000.0,    # Very high weight to keep feet pinned
                200.0,
            ))

    # Hand constraints at start and end
    for i in range(len(hand_link_indices)):
        # Start constraint
        hand_start_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(hand_start_wxyzs[i]),
                                                                   hand_start_positions[i])
        factors.append(
            pk.costs.pose_cost_with_base(
                robot,    # Unbatched robot for single timestep
                robot.joint_var_cls(0),
                jaxls.SE3Var(0),
                hand_start_pose,
                jnp.array(hand_link_indices[i]),
                100.0,
                20.0,
            ))

        # End constraint
        hand_end_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(hand_end_wxyzs[i]),
                                                                 hand_end_positions[i])
        factors.append(
            pk.costs.pose_cost_with_base(
                robot,    # Unbatched robot for single timestep
                robot.joint_var_cls(1),
                jaxls.SE3Var(1),
                hand_end_pose,
                jnp.array(hand_link_indices[i]),
                100.0,
                20.0,
            ))

    # Smoothness costs for joints
    factors.extend([
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.1])[None],
        ),
    ])

    # Add rest cost - use custom version if rest poses are provided
    if rest_joint_pose is not None or rest_base_pose is not None:
        # Use custom rest cost with specified poses
        actual_rest_joint_pose = rest_joint_pose if rest_joint_pose is not None else jnp.array(
            joint_vars.default_factory()[0])

        # Convert rest_base_pose to SE3 if provided
        if rest_base_pose is not None:
            rest_base_SE3 = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.array(rest_base_pose[1])),    # wxyz
                jnp.array(rest_base_pose[0])    # position
            )
        else:
            rest_base_SE3 = jaxlie.SE3.identity()

        # Apply rest cost to each timestep
        # Note: We need to broadcast the rest poses to match the batch dimension
        rest_base_SE3_batched = jax.tree.map(lambda x: jnp.repeat(x[None], timesteps, axis=0),
                                             rest_base_SE3)

        factors.append(
            pk.costs.rest_with_base_cost_custom(
                joint_vars,
                base_vars,
                jnp.broadcast_to(
                    actual_rest_joint_pose[None],
                    (timesteps, len(actual_rest_joint_pose))),    # Broadcast to all timesteps
                rest_base_SE3_batched,    # Pass SE3 object, not wxyz_xyz
                jnp.broadcast_to(
                    jnp.array([0.01] * robot.joints.num_actuated_joints + [0.1] * 3 +
                              [0.001] * 3)[None],
                    (timesteps, robot.joints.num_actuated_joints + 6)),    # Broadcast weights
            ))
    else:
        # Use default rest cost (current behavior)
        factors.extend([
            pk.costs.rest_cost(
                joint_vars,
                joint_vars.default_factory()[None],
                jnp.array([0.01])[None],
            ),
        ])

    # Add limit cost
    factors.append(pk.costs.limit_cost(robot_batched, joint_vars, jnp.array([100.0])[None]))

    # Base smoothness
    @jaxls.Cost.create_factory(name="BaseSmoothnessCost")
    def base_smoothness_cost(vals, prev_vars, curr_vars):
        prev_poses = vals[prev_vars]
        curr_poses = vals[curr_vars]
        delta = (prev_poses.inverse() @ curr_poses).log()
        return (delta * 1.0).flatten()

    factors.append(
        base_smoothness_cost(
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
            jaxls.SE3Var(jnp.arange(1, timesteps)),
        ))

    # Add COM support polygon cost (weight controls whether it has any effect)
    # Get foot link indices for COM support
    foot_link_names_for_com = robots_config.get_foot_link_names(robot.name)
    foot_link_indices_for_com = jnp.array(
        [robot.links.names.index(name) for name in foot_link_names_for_com])

    # Add COM support cost for all timesteps (batched)
    factors.append(
        pk.costs.com_support_polygon_cost_with_base(
            robot_batched,
            joint_vars,
            base_vars,
            jnp.broadcast_to(foot_link_indices_for_com[None, :], (timesteps,) +
                             foot_link_indices_for_com.shape),    # Broadcast to all timesteps
            num_directions=8,
            weight=com_support_weight,
            margin_threshold=com_support_margin,
        ))

    # Solve
    sol = (jaxls.LeastSquaresProblem(factors, [joint_vars, base_vars]).analyze().solve(
        initial_vals=jaxls.VarValues.make([
            joint_vars.with_value(init_joint_traj),
            base_vars.with_value(init_base_poses),
        ]),
        verbose=False,
    ))

    return sol[base_vars], sol[joint_vars]
