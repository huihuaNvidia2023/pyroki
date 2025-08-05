"""Data structures for PyRoKi logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime


@dataclass
class FrameData:
    """Data for a single frame/timestep.
    
    All arrays should be numpy arrays for efficient storage and computation.
    """
    # Timestamp relative to episode start (seconds)
    timestamp: float

    # Joint data
    joint_names: List[str]
    joint_positions: np.ndarray    # Shape: (N,)
    joint_velocities: np.ndarray    # Shape: (N,)
    joint_accelerations: np.ndarray    # Shape: (N,)

    # Link data
    link_names: List[str]
    link_positions: np.ndarray    # Shape: (M, 3)
    link_rotations: np.ndarray    # Shape: (M, 4) - quaternions (w, x, y, z)
    link_linear_velocities: np.ndarray    # Shape: (M, 3)
    link_angular_velocities: np.ndarray    # Shape: (M, 3)

    # Root/base data
    root_position: np.ndarray    # Shape: (3,)
    root_quaternion: np.ndarray    # Shape: (4,) - (w, x, y, z)
    root_linear_velocity: np.ndarray    # Shape: (3,)
    root_angular_velocity: np.ndarray    # Shape: (3,)

    # Contact status
    contact_status: np.ndarray    # Shape: (M,) - bool array

    def get_joint_position(self, joint_name: str) -> float:
        """Get the position of a specific joint by name.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Position value for the joint
            
        Raises:
            ValueError: If joint name not found
        """
        try:
            idx = self.joint_names.index(joint_name)
            return float(self.joint_positions[idx])
        except ValueError:
            raise ValueError(f"Joint '{joint_name}' not found in frame. Available joints: {self.joint_names}")
    
    def get_link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the position and rotation of a specific link by name.
        
        Args:
            link_name: Name of the link
            
        Returns:
            Tuple of (position, quaternion) for the link
            
        Raises:
            ValueError: If link name not found
        """
        try:
            idx = self.link_names.index(link_name)
            return self.link_positions[idx], self.link_rotations[idx]
        except ValueError:
            raise ValueError(f"Link '{link_name}' not found in frame. Available links: {self.link_names}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parquet storage."""
        return {
            'timestamp': self.timestamp,
            'joint_names': self.joint_names,
            'joint_positions': self.joint_positions.tolist(),
            'joint_velocities': self.joint_velocities.tolist(),
            'joint_accelerations': self.joint_accelerations.tolist(),
            'link_names': self.link_names,
            'link_positions': self.link_positions.flatten().tolist(),    # Flatten for storage
            'link_rotations': self.link_rotations.flatten().tolist(),
            'link_linear_velocities': self.link_linear_velocities.flatten().tolist(),
            'link_angular_velocities': self.link_angular_velocities.flatten().tolist(),
            'root_position': self.root_position.tolist(),
            'root_quaternion': self.root_quaternion.tolist(),
            'root_linear_velocity': self.root_linear_velocity.tolist(),
            'root_angular_velocity': self.root_angular_velocity.tolist(),
            'contact_status': self.contact_status.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], num_joints: int, num_links: int) -> FrameData:
        """Reconstruct from dictionary loaded from parquet."""
        return cls(
            timestamp=data['timestamp'],
            joint_names=data['joint_names'],
            joint_positions=np.array(data['joint_positions']),
            joint_velocities=np.array(data['joint_velocities']),
            joint_accelerations=np.array(data['joint_accelerations']),
            link_names=data['link_names'],
            link_positions=np.array(data['link_positions']).reshape(num_links, 3),
            link_rotations=np.array(data['link_rotations']).reshape(num_links, 4),
            link_linear_velocities=np.array(data['link_linear_velocities']).reshape(num_links, 3),
            link_angular_velocities=np.array(data['link_angular_velocities']).reshape(num_links, 3),
            root_position=np.array(data['root_position']),
            root_quaternion=np.array(data['root_quaternion']),
            root_linear_velocity=np.array(data['root_linear_velocity']),
            root_angular_velocity=np.array(data['root_angular_velocity']),
            contact_status=np.array(data['contact_status'], dtype=bool),
        )


@dataclass
class EpisodeMetadata:
    """Metadata for an episode.
    
    This is designed to be easily extensible - users can add custom fields.
    """
    # Required fields
    robot_name: str
    episode_id: str
    start_time: datetime
    duration: float    # Total duration in seconds
    num_frames: int

    # Recording configuration
    recording_config: Dict[str, Any] = field(default_factory=dict)

    # User-extensible metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_custom_field(self, key: str, value: Any):
        """Add a custom metadata field."""
        self.custom_metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'robot_name': self.robot_name,
            'episode_id': self.episode_id,
            'start_time': self.start_time.isoformat(),
            'duration': self.duration,
            'num_frames': self.num_frames,
            'recording_config': self.recording_config,
            'custom_metadata': self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeMetadata:
        """Reconstruct from dictionary."""
        return cls(
            robot_name=data['robot_name'],
            episode_id=data['episode_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            duration=data['duration'],
            num_frames=data['num_frames'],
            recording_config=data.get('recording_config', {}),
            custom_metadata=data.get('custom_metadata', {}),
        )


@dataclass
class EpisodeData:
    """Complete data for an episode (single frame or trajectory)."""
    metadata: EpisodeMetadata
    frames: List[FrameData]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> FrameData:
        return self.frames[idx]

    def add_frame(self, frame: FrameData):
        """Add a frame to the episode."""
        self.frames.append(frame)
        self.metadata.num_frames = len(self.frames)
        if self.frames:
            self.metadata.duration = self.frames[-1].timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        # Store metadata separately
        result = {
            'metadata': self.metadata.to_dict(),
            'num_joints': len(self.frames[0].joint_names) if self.frames else 0,
            'num_links': len(self.frames[0].link_names) if self.frames else 0,
        }

        # Store frame data as lists of values
        if self.frames:
            # Extract all frame data into columnar format for efficient storage
            frame_dicts = [frame.to_dict() for frame in self.frames]

            # Convert to columnar format
            for key in frame_dicts[0].keys():
                result[key] = [frame[key] for frame in frame_dicts]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeData:
        """Reconstruct from dictionary."""
        metadata = EpisodeMetadata.from_dict(data['metadata'])

        # Reconstruct frames
        frames = []
        num_frames = metadata.num_frames
        num_joints = data['num_joints']
        num_links = data['num_links']

        if num_frames > 0:
            # Convert from columnar format back to frames
            for i in range(num_frames):
                frame_dict = {}
                for key in [
                        'timestamp', 'joint_names', 'joint_positions', 'joint_velocities',
                        'joint_accelerations', 'link_names', 'link_positions', 'link_rotations',
                        'link_linear_velocities', 'link_angular_velocities', 'root_position',
                        'root_quaternion', 'root_linear_velocity', 'root_angular_velocity',
                        'contact_status'
                ]:
                    if key in data:
                        frame_dict[key] = data[key][i]

                frames.append(FrameData.from_dict(frame_dict, num_joints, num_links))

        return cls(metadata=metadata, frames=frames)
