"""
Example of using rest_with_base_cost_custom in IK solving.
This is a modified version of _solve_ik_with_base that uses custom rest poses.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk


@jdc.jit
def solve_ik_with_base_custom_rest(
        robot: pk.Robot,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        target_joint_indices: jax.Array,
        fix_base: jnp.ndarray,
        prev_pos: jnp.ndarray,
        prev_wxyz: jnp.ndarray,
        prev_cfg: jnp.ndarray,
        rest_joint_pose: jnp.ndarray,    # Custom joint rest pose
        rest_base_pose: jaxlie.SE3,    # Custom base rest pose
) -> tuple[jaxlie.SE3, jax.Array]:
    """Solve IK with custom rest poses for both joints and base."""

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
    ):
        ...

    base_var = ConstrainedSE3Var(0)
    joint_var = JointVar(0)

    # Get the batch axes for the variable through the target pose.
    target_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
    batch_axes = target_pose.get_batch_axes()

    factors = [
        pk.costs.pose_cost_with_base_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            ConstrainedSE3Var(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=jnp.array([50.0] * len(target_joint_indices)),
            ori_weight=jnp.array([10.0] * len(target_joint_indices)),
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            jnp.array([100.0] * robot.joints.num_joints),
        ),
    # Use the new custom rest cost with specified rest poses
        pk.costs.rest_with_base_cost_custom(
            joint_var,
            base_var,
            rest_joint_pose,    # Custom joint rest configuration
            rest_base_pose,    # Custom base rest pose (not just identity!)
            jnp.array([0.01] * robot.joints.num_actuated_joints    # Joint weights
                      + [0.1] * 3    # Base position weights
                      + [0.001] * 3    # Base orientation weights
                     ),
        ),
    ]

    sol = (jaxls.LeastSquaresProblem(factors, [joint_var, base_var]).analyze().solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg), base_var]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
    ))
    return sol[base_var], sol[joint_var]


# Example usage:
def example_usage():
    """Shows how to use the custom rest pose IK solver."""
    # ... robot setup ...

    # Define custom rest poses
    rest_joint_config = jnp.zeros(robot.joints.num_actuated_joints)    # or any desired config
    rest_base_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_quaternion_wxyz(jnp.array([1.0, 0.0, 0.0, 0.0])),
        jnp.array([0.0, 0.0, 0.75])    # Rest at 0.75m height instead of origin
    )

    # Solve IK with custom rest poses
    base_pose, joint_config = solve_ik_with_base_custom_rest(
        robot=robot,
        target_wxyz=target_wxyz,
        target_position=target_position,
        target_joint_indices=target_indices,
        fix_base=fix_base,
        prev_pos=prev_pos,
        prev_wxyz=prev_wxyz,
        prev_cfg=prev_cfg,
        rest_joint_pose=rest_joint_config,
        rest_base_pose=rest_base_pose,
    )
