"""Trajectory state management for interactive trajectory optimization."""

from typing import Dict, Tuple, Optional
import numpy as np
from enum import Enum, auto


class TrajectoryStates(Enum):
    """Enumeration of possible trajectory planning states."""
    SETUP_START = auto()
    SETUP_END = auto()
    READY = auto()
    OPTIMIZING = auto()
    PLAYBACK = auto()


class TrajectoryState:
    """Manages the state of interactive trajectory planning."""
    
    def __init__(self):
        self.current_state = TrajectoryStates.SETUP_START
        
        # Captured poses
        self.start_positions: Dict[str, np.ndarray] = {}
        self.start_wxyzs: Dict[str, np.ndarray] = {}
        self.end_positions: Dict[str, np.ndarray] = {}
        self.end_wxyzs: Dict[str, np.ndarray] = {}
        
        # Trajectory data after optimization
        self.base_positions: Optional[np.ndarray] = None
        self.base_wxyzs: Optional[np.ndarray] = None
        self.joint_cfgs: Optional[np.ndarray] = None
        self.timesteps: int = 30
        self.dt: float = 0.05
        
    def capture_start_poses(self, positions: Dict[str, np.ndarray], 
                           wxyzs: Dict[str, np.ndarray]) -> None:
        """Capture the start poses for all links."""
        self.start_positions = positions.copy()
        self.start_wxyzs = wxyzs.copy()
        self.current_state = TrajectoryStates.SETUP_END
        
    def capture_end_poses(self, positions: Dict[str, np.ndarray],
                         wxyzs: Dict[str, np.ndarray]) -> None:
        """Capture the end poses for all links."""
        self.end_positions = positions.copy()
        self.end_wxyzs = wxyzs.copy()
        self.current_state = TrajectoryStates.READY
        
    def set_trajectory_data(self, base_positions: np.ndarray,
                           base_wxyzs: np.ndarray,
                           joint_cfgs: np.ndarray) -> None:
        """Store the optimized trajectory data."""
        self.base_positions = base_positions
        self.base_wxyzs = base_wxyzs
        self.joint_cfgs = joint_cfgs
        self.current_state = TrajectoryStates.PLAYBACK
        
    def reset(self) -> None:
        """Reset to initial state."""
        self.__init__()
        
    def get_state_text(self) -> str:
        """Get human-readable state description."""
        state_texts = {
            TrajectoryStates.SETUP_START: "Setup Start Pose",
            TrajectoryStates.SETUP_END: "Setup End Pose",
            TrajectoryStates.READY: "Ready to Optimize",
            TrajectoryStates.OPTIMIZING: "Optimizing...",
            TrajectoryStates.PLAYBACK: "Playing Trajectory"
        }
        return state_texts[self.current_state] 