"""Robot-specific configurations for PyRoKi.

This module stores robot-specific parameters like foot dimensions,
support polygon configurations, etc. to ensure consistency across the codebase.
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any

# Robot foot dimensions (length, width) in meters
ROBOT_FOOT_DIMENSIONS: Dict[str, Tuple[float, float]] = {
    "g1_description": (0.2, 0.06),    # G1 humanoid: 15cm x 8cm feet
    "panda_description": (0.1, 0.1),    # Panda (for testing): 10cm x 10cm
    # Add more robots here as needed
}

# Foot length offset ratio - shifts the support polygon forward/backward
# Positive values shift the polygon forward (towards toes)
FOOT_LENGTH_OFFSET_RATIO: Dict[str, float] = {
    "g1_description": 0.175,    # No offset for G1
    "panda_description": 0.0,    # No offset for Panda
    # Add more robots here as needed
}


def get_foot_dimensions(robot_description: str) -> Tuple[float, float]:
    """Get foot dimensions for a given robot.
    
    Args:
        robot_description: Robot description name (e.g., "g1_description")
        
    Returns:
        Tuple of (length, width) in meters
    """
    if robot_description not in ROBOT_FOOT_DIMENSIONS:
        raise ValueError(f"Unknown robot description: {robot_description}. "
                         f"Available: {list(ROBOT_FOOT_DIMENSIONS.keys())}")
    return ROBOT_FOOT_DIMENSIONS[robot_description]


def get_foot_length_offset_ratio(robot_description: str) -> float:
    """Get foot length offset ratio for a given robot.
    
    Args:
        robot_description: Robot description name (e.g., "g1_description")
        
    Returns:
        Offset ratio (0.0 = centered, positive = forward shift)
    """
    if robot_description not in FOOT_LENGTH_OFFSET_RATIO:
        return 0.0    # Default to no offset
    return FOOT_LENGTH_OFFSET_RATIO[robot_description]


def compute_foot_local_corners(foot_length: float = None,
                               foot_width: float = None,
                               length_offset_ratio: float = None,
                               robot_description: str = None) -> jnp.ndarray:
    """Compute local foot corner positions.
    
    This function can be called in two ways:
    1. With explicit parameters: compute_foot_local_corners(foot_length, foot_width, length_offset_ratio)
    2. With robot description: compute_foot_local_corners(robot_description="g1_description")
    
    Args:
        foot_length: Length of the foot (along X-axis) in meters
        foot_width: Width of the foot (along Y-axis) in meters  
        length_offset_ratio: Ratio to shift support polygon forward/backward
                           (0.0 = centered, positive = forward)
        robot_description: Robot description name to get all parameters from config
    
    Returns:
        Array of shape (4, 3) with corner positions in local foot frame
        Order: front-right, front-left, back-left, back-right
    """
    # If robot_description is provided, get all parameters from config
    if robot_description is not None:
        foot_length, foot_width = get_foot_dimensions(robot_description)
        length_offset_ratio = get_foot_length_offset_ratio(robot_description)
    else:
        # Ensure required parameters are provided
        if foot_length is None or foot_width is None:
            raise ValueError("Either provide robot_description or (foot_length, foot_width)")
        if length_offset_ratio is None:
            length_offset_ratio = 0.0

    L, W = foot_length, foot_width
    offset = length_offset_ratio * L

    local_corners = jnp.array([
        [L / 2 + offset, W / 2, 0],    # Front-right
        [L / 2 + offset, -W / 2, 0],    # Front-left
        [-L / 2 + offset, -W / 2, 0],    # Back-left
        [-L / 2 + offset, W / 2, 0],    # Back-right
    ])

    return local_corners


def get_foot_link_names(robot_description: str) -> Tuple[str, str]:
    """Get foot link names for a given robot.
    
    Args:
        robot_description: Robot description name (e.g., "g1_description")
        
    Returns:
        Tuple of (left_foot_link, right_foot_link) names
    """
    foot_link_names = {
        "g1_description": ("left_ankle_roll_link", "right_ankle_roll_link"),
    # Add more robots here as needed
    }

    if robot_description not in foot_link_names:
        raise ValueError(f"Unknown robot description: {robot_description}. "
                         f"Available: {list(foot_link_names.keys())}")
    return foot_link_names[robot_description]
