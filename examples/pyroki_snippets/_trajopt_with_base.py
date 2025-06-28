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
        foot_link_indices=jnp.array(foot_link_indices),
        hand_link_indices=jnp.array(hand_link_indices),
        foot_positions=foot_positions,
        foot_wxyzs=foot_wxyzs,
        hand_start_positions=hand_start_positions,
        hand_start_wxyzs=hand_start_wxyzs,
        hand_end_positions=hand_end_positions,
        hand_end_wxyzs=hand_end_wxyzs,
        fix_base=jnp.array(fix_base_position + fix_base_orientation),
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
    )

    # Extract results
    base_positions = np.array(base_poses.translation())
    base_wxyzs = np.array(base_poses.rotation().wxyz)
    joint_cfgs = np.array(joint_traj)

    return base_positions, base_wxyzs, joint_cfgs


@jdc.jit
def _solve_start_end_iks(
    robot: pk.Robot,
    foot_link_indices: jax.Array,
    hand_link_indices: jax.Array,
    foot_positions: jax.Array,
    foot_wxyzs: jax.Array,
    hand_start_positions: jax.Array,
    hand_start_wxyzs: jax.Array,
    hand_end_positions: jax.Array,
    hand_end_wxyzs: jax.Array,
    fix_base: jax.Array,
) -> Tuple[jaxlie.SE3, jax.Array, jaxlie.SE3, jax.Array]:
    """Solve IK for start and end configurations."""

    # Create variables
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)

    # Use standard SE3Var for now
    base_var_0 = jaxls.SE3Var(0)
    base_var_1 = jaxls.SE3Var(1)

    factors = []

    # Foot constraints (same for both start and end)
    for i in range(len(foot_link_indices)):
        foot_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(foot_wxyzs[i]),
                                                             foot_positions[i])
        # Start
        factors.append(
            pk.costs.pose_cost_with_base(robot, joint_var_0, base_var_0, foot_pose,
                                         foot_link_indices[i], 500.0, 100.0))
        # End
        factors.append(
            pk.costs.pose_cost_with_base(robot, joint_var_1, base_var_1, foot_pose,
                                         foot_link_indices[i], 500.0, 100.0))

    # Hand constraints
    for i in range(len(hand_link_indices)):
        # Start
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

        # End
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

    # Standard costs
    factors.extend([
        pk.costs.limit_cost(robot, joint_var_0, 100.0),
        pk.costs.limit_cost(robot, joint_var_1, 100.0),
        pk.costs.rest_cost(joint_var_0, joint_var_0.default_factory(), 0.01),
        pk.costs.rest_cost(joint_var_1, joint_var_1.default_factory(), 0.01),
    ])

    sol = (jaxls.LeastSquaresProblem(
        factors, [joint_var_0, joint_var_1, base_var_0, base_var_1]).analyze().solve(verbose=False))

    return sol[base_var_0], sol[joint_var_0], sol[base_var_1], sol[joint_var_1]


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
        pk.costs.rest_cost(
            joint_vars,
            joint_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.limit_cost(robot_batched, joint_vars,
                            jnp.array([100.0])[None]),
    ])

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

    # Solve
    sol = (jaxls.LeastSquaresProblem(factors, [joint_vars, base_vars]).analyze().solve(
        initial_vals=jaxls.VarValues.make([
            joint_vars.with_value(init_joint_traj),
            base_vars.with_value(init_base_poses),
        ]),
        verbose=False,
    ))

    return sol[base_vars], sol[joint_vars]
