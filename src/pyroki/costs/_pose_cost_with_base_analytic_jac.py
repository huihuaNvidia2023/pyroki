from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot
from ._pose_cost_analytic_jac import _get_actuated_joints_applied_to_target

# Cache includes T_world_base for base transform jacobian computation
_PoseCostWithBaseJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3, jaxlie.SE3]


def pose_cost_with_base_analytic_jac(
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
) -> jaxls.Cost:
    # We only check shape lengths because there might be (1,) axes for
    # broadcasting reasons.
    assert (len(target_link_index.shape) == len(jnp.asarray(joint_var.id).shape) == len(
        robot.joints.twists.shape[:-2])), "Batch axes of inputs should match"

    # Broadcast the inputs for _get_actuated_joints_applied_to_target().
    # Excluding the weights for now...
    batch_axes = jnp.broadcast_shapes(
        target_pose.get_batch_axes(),
        jnp.asarray(joint_var.id).shape,
        target_pose.get_batch_axes(),
        target_link_index.shape,
    )
    broadcast_batch_axes = partial(
        jax.tree.map,
        lambda x: jnp.broadcast_to(x, batch_axes + x.shape[len(batch_axes):]),
    )
    get_actuated_joints = _get_actuated_joints_applied_to_target
    for _ in range(len(batch_axes)):
        get_actuated_joints = jax.vmap(get_actuated_joints)

    # Compute applied joints.
    robot = broadcast_batch_axes(robot)
    base_link_mask = robot.links.parent_joint_indices == -1
    parent_joint_indices = jnp.where(base_link_mask, 0, robot.links.parent_joint_indices)
    target_joint_idx = parent_joint_indices[tuple(
        jnp.arange(d) for d in parent_joint_indices.shape[:-1]) + (target_link_index,)]
    joints_applied_to_target = get_actuated_joints(broadcast_batch_axes(robot),
                                                   broadcast_batch_axes(target_joint_idx))

    return _pose_cost_with_base_analytical_jac(
        robot,
        joint_var,
        T_world_base_var,
        target_pose,
        target_link_index,
        pos_weight,
        ori_weight,
        joints_applied_to_target,
    )


# It's nice to pass arguments in explicitly instead of via closure in the
# `pose_cost_with_base_analytic_jac` wrapper. It helps jaxls vectorize repeated costs.
def _pose_cost_with_base_jac(
    vals: jaxls.VarValues,
    jac_cache: _PoseCostWithBaseJacCache,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
) -> jax.Array:
    """Jacobian for pose cost with base transform and analytic computation."""
    del vals, target_pose    # Unused!
    Ts_base_joint, Ts_base_link, pose_error, T_world_base = jac_cache

    T_base_ee = jaxlie.SE3(Ts_base_link[target_link_index])
    T_world_ee = T_world_base @ T_base_ee
    Ts_world_joint = T_world_base @ jaxlie.SE3(Ts_base_joint)

    R_ee_world = T_world_ee.rotation().inverse()

    # Get joint twists; these are scaled for mimic joints.
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]

    # Get angular velocity components (omega).
    omega_local = joint_twists[:, 3:]
    omega_wrt_world = Ts_world_joint.rotation() @ omega_local
    omega_wrt_ee = R_ee_world @ omega_wrt_world

    # Get linear velocity components (v).
    vel_local = joint_twists[:, :3]
    vel_wrt_world = Ts_world_joint.rotation() @ vel_local

    # Compute the linear velocity component (v = ω × r + v_joint).
    vel_wrt_world = (jnp.cross(
        omega_wrt_world,
        T_world_ee.translation() - Ts_world_joint.translation(),
    ) + vel_wrt_world)
    vel_wrt_ee = R_ee_world @ vel_wrt_world

    # Combine into spatial Jacobian for joints.
    jac_joints = jnp.where(
        joints_applied_to_target[:, None] != -1,
        jnp.concatenate(
            [
                vel_wrt_ee,
                omega_wrt_ee,
            ],
            axis=1,
        ),
        0.0,
    ).T
    jac_joints = pose_error.jlog() @ jac_joints

    # Jacobian of all joints => Jacobian of actuated joints.
    #
    # Because of mimic joints, the Jacobian terms from multiple joints can be
    # applied to a single actuated joint. This is summed!
    jac_joints = (jnp.zeros(
        (6, robot.joints.num_actuated_joints)).at[:, joints_applied_to_target].add(
            (joints_applied_to_target[None, :] != -1) * jac_joints))

    # Compute base transform Jacobian
    # The end-effector velocity due to base motion is the adjoint transform
    # For SE3, J_base = Ad_{T_base_ee^{-1}}
    T_ee_base = T_base_ee.inverse()
    jac_base = pose_error.jlog() @ T_ee_base.adjoint()

    # Apply weights
    weights = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    jac_joints = jac_joints * weights[:, None]
    jac_base = jac_base * weights[:, None]

    # Return concatenated jacobian for both variables
    # The order must match the order of variables in the problem
    return jnp.concatenate([jac_joints, jac_base], axis=1)


@jaxls.Cost.create_factory(jac_custom_with_cache_fn=_pose_cost_with_base_jac)
def _pose_cost_with_base_analytical_jac(
    vals: jaxls.VarValues,
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
    joints_applied_to_target: jax.Array,
) -> tuple[jax.Array, _PoseCostWithBaseJacCache]:
    """Computes the residual for matching link poses to target poses with mobile base."""
    del joints_applied_to_target
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    T_world_base = vals[T_world_base_var]

    # FK returns poses in base frame
    Ts_base_joint = robot._forward_kinematics_joints(joint_cfg)
    Ts_base_link = robot._link_poses_from_joint_poses(Ts_base_joint)

    # Transform to world frame
    T_base_ee = jaxlie.SE3(Ts_base_link[target_link_index, :])
    T_world_ee = T_world_base @ T_base_ee

    pose_error = target_pose.inverse() @ T_world_ee
    return (
        pose_error.log() * jnp.array([pos_weight] * 3 + [ori_weight] * 3),
    # Cache: base-frame transforms and world-base transform for jacobian
        (Ts_base_joint, Ts_base_link, pose_error, T_world_base),
    )
