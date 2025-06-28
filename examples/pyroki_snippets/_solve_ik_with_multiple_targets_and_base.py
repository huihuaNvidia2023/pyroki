"""
Solves the basic IK problem.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_with_multiple_targets_and_base(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
    fix_base_position: tuple[bool, bool, bool],
    fix_base_orientation: tuple[bool, bool, bool],
    prev_pos: onp.ndarray,
    prev_wxyz: onp.ndarray,
    prev_cfg: onp.ndarray,
    pos_weights: onp.ndarray | None = None,
    ori_weights: onp.ndarray | None = None,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.
        fix_base_position: Whether to fix the base position (x, y, z).
        fix_base_orientation: Whether to fix the base orientation (w_x, w_y, w_z).
        prev_pos, prev_wxyz, prev_cfg: Previous base position, orientation, and joint configuration, for smooth motion.
        pos_weights: onp.ndarray. Shape: (num_targets,). Position weights for each target. If None, uses default 50.0.
        ori_weights: onp.ndarray. Shape: (num_targets,). Orientation weights for each target. If None, uses default 10.0.
    Returns:
        base_pos: onp.ndarray. Shape: (3,).
        base_wxyz: onp.ndarray. Shape: (4,).
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    assert prev_pos.shape == (3,) and prev_wxyz.shape == (4,)
    assert prev_cfg.shape == (robot.joints.num_actuated_joints,)
    
    # Set default weights if not provided
    if pos_weights is None:
        pos_weights = onp.full(num_targets, 50.0)
    if ori_weights is None:
        ori_weights = onp.full(num_targets, 10.0)
    
    assert pos_weights.shape == (num_targets,)
    assert ori_weights.shape == (num_targets,)
    
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    base_pose, cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
        jnp.array(fix_base_position + fix_base_orientation),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),
        jnp.array(pos_weights),
        jnp.array(ori_weights),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    base_pos = base_pose.translation()
    base_wxyz = base_pose.rotation().wxyz
    assert base_pos.shape == (3,) and base_wxyz.shape == (4,)

    return onp.array(base_pos), onp.array(base_wxyz), onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
    fix_base: jnp.ndarray,
    prev_pos: jnp.ndarray,
    prev_wxyz: jnp.ndarray,
    prev_cfg: jnp.ndarray,
    pos_weights: jax.Array,
    ori_weights: jax.Array,
) -> tuple[jaxlie.SE3, jax.Array]:
    JointVar = robot.joint_var_cls
  
    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - fix_base)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(prev_wxyz),
            prev_pos,
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    base_var = ConstrainedSE3Var(0)
    joint_var = JointVar(0)
    
    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    batch_axes = target_pose.get_batch_axes()

    factors = [
        pk.costs.pose_cost_with_base_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            ConstrainedSE3Var(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=pos_weights,
            ori_weight=ori_weights,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            jnp.array([100.0] * robot.joints.num_joints),
        ),
        pk.costs.rest_with_base_cost(
            joint_var,
            base_var,
            jnp.array(joint_var.default_factory()),
            jnp.array(
                [0.01] * robot.joints.num_actuated_joints
                + [0.1] * 3  # Base position DoF.
                + [0.001] * 3,  # Base orientation DoF.
            ),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var, base_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(prev_cfg), base_var]
            ),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[base_var], sol[joint_var]
