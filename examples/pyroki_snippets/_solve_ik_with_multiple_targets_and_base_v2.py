"""
Solves IK with separate handling for foot and hand constraints.
Feet are kept in world frame with high weights, hands are solved relative to base.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_with_multiple_targets_and_base_v2(
    robot: pk.Robot,
    foot_link_names: Sequence[str],
    hand_link_names: Sequence[str],
    foot_positions: onp.ndarray,
    foot_wxyzs: onp.ndarray,
    hand_positions: onp.ndarray,
    hand_wxyzs: onp.ndarray,
    fix_base_position: tuple[bool, bool, bool],
    fix_base_orientation: tuple[bool, bool, bool],
    prev_pos: onp.ndarray,
    prev_wxyz: onp.ndarray,
    prev_cfg: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Solves IK with separate handling for feet and hands.
    
    This version treats feet and hands differently:
    - Feet: High-weight world-frame constraints (essentially pinned)
    - Hands: Normal-weight constraints that can be satisfied by base + arm motion
    
    Args:
        robot: PyRoKi Robot.
        foot_link_names: List of foot link names (e.g., ankle links).
        hand_link_names: List of hand link names.
        foot_positions: Shape: (num_feet, 3). World-frame foot positions.
        foot_wxyzs: Shape: (num_feet, 4). World-frame foot orientations.
        hand_positions: Shape: (num_hands, 3). World-frame hand positions.
        hand_wxyzs: Shape: (num_hands, 4). World-frame hand orientations.
        fix_base_position: Whether to fix the base position (x, y, z).
        fix_base_orientation: Whether to fix the base orientation (roll, pitch, yaw).
        prev_pos, prev_wxyz, prev_cfg: Previous base position, orientation, and joint configuration.
    
    Returns:
        base_pos: Shape: (3,).
        base_wxyz: Shape: (4,).
        cfg: Shape: (robot.joints.num_actuated_joints,).
    """
    num_feet = len(foot_link_names)
    num_hands = len(hand_link_names)
    
    assert foot_positions.shape == (num_feet, 3)
    assert foot_wxyzs.shape == (num_feet, 4)
    assert hand_positions.shape == (num_hands, 3)
    assert hand_wxyzs.shape == (num_hands, 4)
    assert prev_pos.shape == (3,) and prev_wxyz.shape == (4,)
    assert prev_cfg.shape == (robot.joints.num_actuated_joints,)
    
    foot_link_indices = [robot.links.names.index(name) for name in foot_link_names]
    hand_link_indices = [robot.links.names.index(name) for name in hand_link_names]

    base_pose, cfg = _solve_ik_jax_v2(
        robot,
        jnp.array(foot_wxyzs),
        jnp.array(foot_positions),
        jnp.array(foot_link_indices),
        jnp.array(hand_wxyzs),
        jnp.array(hand_positions),
        jnp.array(hand_link_indices),
        jnp.array(fix_base_position + fix_base_orientation),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    base_pos = base_pose.translation()
    base_wxyz = base_pose.rotation().wxyz
    assert base_pos.shape == (3,) and base_wxyz.shape == (4,)

    return onp.array(base_pos), onp.array(base_wxyz), onp.array(cfg)


@jdc.jit
def _solve_ik_jax_v2(
    robot: pk.Robot,
    foot_wxyz: jax.Array,
    foot_position: jax.Array,
    foot_joint_indices: jax.Array,
    hand_wxyz: jax.Array,
    hand_position: jax.Array,
    hand_joint_indices: jax.Array,
    fix_base: jnp.ndarray,
    prev_pos: jnp.ndarray,
    prev_wxyz: jnp.ndarray,
    prev_cfg: jnp.ndarray,
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
    
    # Create target poses
    foot_target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(foot_wxyz), foot_position
    )
    hand_target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(hand_wxyz), hand_position
    )
    
    foot_batch_axes = foot_target_pose.get_batch_axes()
    hand_batch_axes = hand_target_pose.get_batch_axes()

    factors = []
    
    # Foot constraints - very high weight to keep them pinned
    if len(foot_joint_indices) > 0:
        factors.append(
            pk.costs.pose_cost_with_base_analytic_jac(
                jax.tree.map(lambda x: x[None], robot),
                JointVar(jnp.full(foot_batch_axes, 0)),
                ConstrainedSE3Var(jnp.full(foot_batch_axes, 0)),
                foot_target_pose,
                foot_joint_indices,
                pos_weight=500.0,  # 10x higher than hands
                ori_weight=100.0,   # 10x higher than hands
            )
        )
    
    # Hand constraints - normal weight for flexibility
    if len(hand_joint_indices) > 0:
        factors.append(
            pk.costs.pose_cost_with_base_analytic_jac(
                jax.tree.map(lambda x: x[None], robot),
                JointVar(jnp.full(hand_batch_axes, 0)),
                ConstrainedSE3Var(jnp.full(hand_batch_axes, 0)),
                hand_target_pose,
                hand_joint_indices,
                pos_weight=50.0,
                ori_weight=10.0,
            )
        )
    
    # Add additional constraints
    factors.extend([
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
    ])
    
    # Add leg joint stiffness to reduce unnecessary leg motion
    # This helps keep the legs stable when only moving hands and base
    # We increase the rest cost weight for leg joints specifically
    leg_joint_patterns = ["hip", "knee", "ankle"]
    leg_joint_indices = []
    for i, name in enumerate(robot.joints.actuated_names):
        if any(pattern in name.lower() for pattern in leg_joint_patterns):
            leg_joint_indices.append(i)
    
    if leg_joint_indices:
        # Create a weight array with higher weights for leg joints
        joint_weights = jnp.ones(robot.joints.num_actuated_joints) * 0.01
        joint_weights = joint_weights.at[jnp.array(leg_joint_indices)].set(1.0)  # 100x stiffer
        
        factors.append(
            pk.costs.rest_cost(
                joint_var,
                rest_pose=prev_cfg,  # Keep legs close to previous configuration
                weight=joint_weights,
            )
        )
    
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