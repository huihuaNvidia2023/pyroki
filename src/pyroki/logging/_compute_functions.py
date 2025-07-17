"""Compute functions for data that isn't directly available from robot state."""

from typing import Callable, Optional, List
import numpy as np
import jax.numpy as jnp
import jaxlie
from .._robot import Robot


class ComputeFunctions:
    """Container for pluggable compute functions.
    
    Users can override these functions to provide custom implementations.
    """

    def __init__(
        self,
        compute_joint_velocity: Optional[Callable] = None,
        compute_joint_acceleration: Optional[Callable] = None,
        compute_link_velocity: Optional[Callable] = None,
        compute_link_angular_velocity: Optional[Callable] = None,
        compute_root_velocity: Optional[Callable] = None,
        compute_root_angular_velocity: Optional[Callable] = None,
        detect_contact: Optional[Callable] = None,
    ):
        """Initialize compute functions.
        
        All functions default to returning zeros if not provided.
        
        Args:
            compute_joint_velocity: Function(robot, joint_positions, prev_positions, dt) -> velocities
            compute_joint_acceleration: Function(robot, joint_velocities, prev_velocities, dt) -> accelerations
            compute_link_velocity: Function(robot, link_positions, prev_positions, dt) -> velocities (M, 3)
            compute_link_angular_velocity: Function(robot, link_positions, **kwargs) -> angular_velocities (M, 3)
                Note: link_positions is passed for shape, but kwargs includes prev_rotations for computation
            compute_root_velocity: Function(position, prev_position, dt) -> velocity (3,)
            compute_root_angular_velocity: Function(rotation, prev_rotation, dt) -> angular_velocity (3,)
            detect_contact: Function(robot, link_positions, link_names, threshold) -> contact_status (M,)
        """
        self.compute_joint_velocity = compute_joint_velocity or self._zeros_joint
        self.compute_joint_acceleration = compute_joint_acceleration or self._zeros_joint
        self.compute_link_velocity = compute_link_velocity or self._zeros_link_vec3
        self.compute_link_angular_velocity = compute_link_angular_velocity or self._zeros_link_vec3
        self.compute_root_velocity = compute_root_velocity or self._zeros_vec3
        self.compute_root_angular_velocity = compute_root_angular_velocity or self._zeros_vec3
        self.detect_contact = detect_contact or default_contact_detector

    @staticmethod
    def _zeros_joint(robot: Robot, joint_positions: np.ndarray, **kwargs) -> np.ndarray:
        """Return zeros for joint data."""
        return np.zeros_like(joint_positions)

    @staticmethod
    def _zeros_link_vec3(robot: Robot, link_positions: np.ndarray, **kwargs) -> np.ndarray:
        """Return zeros for link vector data.
        
        Args:
            robot: Robot instance
            link_positions: Link positions with shape (M, 3) - used for shape reference
            **kwargs: Additional arguments (may include prev_positions, prev_rotations, dt)
        
        Returns:
            Zero array with shape (M, 3)
        """
        return np.zeros_like(link_positions)

    @staticmethod
    def _zeros_vec3(**kwargs) -> np.ndarray:
        """Return zero 3-vector."""
        return np.zeros(3)


def default_contact_detector(robot: Robot,
                             link_positions: np.ndarray,
                             link_names: List[str],
                             threshold: float = 0.05,
                             **kwargs) -> np.ndarray:
    """Default contact detector using height threshold.
    
    Args:
        robot: Robot instance
        link_positions: Link positions in world frame, shape (M, 3)
        link_names: List of link names
        threshold: Height threshold for contact detection
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Contact status array of shape (M,)
    """
    # Simple height-based detection
    return link_positions[:, 2] < threshold


def height_threshold_contact_detector(threshold: float = 0.05) -> Callable:
    """Create a height threshold contact detector with custom threshold.
    
    Args:
        threshold: Height threshold for contact detection
    
    Returns:
        Contact detector function
    """

    def detector(robot: Robot, link_positions: np.ndarray, link_names: List[str],
                 **kwargs) -> np.ndarray:
        return link_positions[:, 2] < threshold

    return detector


def finite_difference_velocity(positions_curr: np.ndarray, positions_prev: np.ndarray,
                               dt: float) -> np.ndarray:
    """Compute velocity using finite differences.
    
    Args:
        positions_curr: Current positions
        positions_prev: Previous positions
        dt: Time step
    
    Returns:
        Velocities
    """
    if dt > 0:
        return (positions_curr - positions_prev) / dt
    else:
        return np.zeros_like(positions_curr)


def finite_difference_acceleration(velocities_curr: np.ndarray, velocities_prev: np.ndarray,
                                   dt: float) -> np.ndarray:
    """Compute acceleration using finite differences.
    
    Args:
        velocities_curr: Current velocities
        velocities_prev: Previous velocities
        dt: Time step
    
    Returns:
        Accelerations
    """
    if dt > 0:
        return (velocities_curr - velocities_prev) / dt
    else:
        return np.zeros_like(velocities_curr)


def finite_difference_angular_velocity(
        rotations_curr: np.ndarray,    # Shape: (4,) quaternion wxyz
        rotations_prev: np.ndarray,    # Shape: (4,) quaternion wxyz
        dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion difference.
    
    Args:
        rotations_curr: Current rotation quaternion (w, x, y, z)
        rotations_prev: Previous rotation quaternion (w, x, y, z)
        dt: Time step
    
    Returns:
        Angular velocity in world frame
    """
    if dt <= 0:
        return np.zeros(3)

    # Convert to jaxlie for quaternion operations
    rot_curr = jaxlie.SO3(jnp.array(rotations_curr))
    rot_prev = jaxlie.SO3(jnp.array(rotations_prev))

    # Compute relative rotation: rot_prev^-1 * rot_curr
    rot_delta = rot_prev.inverse() @ rot_curr

    # Get the log map (axis-angle representation)
    axis_angle = rot_delta.log()

    # Angular velocity is axis-angle / dt in world frame
    # Need to transform from body frame to world frame
    angular_vel_body = np.array(axis_angle) / dt
    angular_vel_world = np.array(rot_curr.apply(jnp.array(angular_vel_body)))

    return angular_vel_world
