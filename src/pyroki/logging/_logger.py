"""Main data logger class for PyRoKi."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp
import jaxlie

from .._robot import Robot
from ._data_structures import FrameData, EpisodeData, EpisodeMetadata
from ._compute_functions import ComputeFunctions


class DataLogger:
    """Main class for logging robot data to parquet files.
    
    Example usage:
        # Create logger
        logger = DataLogger(robot, output_dir="logs")
        
        # Start a new episode
        logger.start_episode()
        
        # Record single frame
        logger.record_frame(joint_cfg, base_pose)
        
        # Or record a trajectory
        for t in range(timesteps):
            logger.record_frame(joint_cfgs[t], base_poses[t], timestamp=t*dt)
        
        # Save episode
        logger.save_episode()
    """

    def __init__(
        self,
        robot: Robot,
        output_dir: Union[str, Path] = "logs",
        compute_functions: Optional[ComputeFunctions] = None,
        contact_threshold: float = 0.05,
        record_velocities: bool = True,
        record_accelerations: bool = True,
        urdf_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the data logger.
        
        Args:
            robot: PyRoKi robot instance
            output_dir: Directory to save log files
            compute_functions: Custom compute functions for velocities/accelerations
            contact_threshold: Height threshold for contact detection
            record_velocities: Whether to record velocities
            record_accelerations: Whether to record accelerations
            urdf_path: Optional path to URDF file (for custom URDFs)
        """
        self.robot = robot
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compute_functions = compute_functions or ComputeFunctions()
        self.contact_threshold = contact_threshold
        self.record_velocities = record_velocities
        self.record_accelerations = record_accelerations
        self.urdf_path = str(urdf_path) if urdf_path else None

        # Current episode data
        self.current_episode: Optional[EpisodeData] = None
        self.episode_start_time: Optional[datetime] = None
        self.episode_counter = self._get_next_episode_number()

        # Previous frame data for velocity/acceleration computation
        self.prev_frame_data: Optional[Dict[str, Any]] = None
        self.prev_timestamp: Optional[float] = None

    def _get_next_episode_number(self) -> int:
        """Get the next episode number based on existing files."""
        existing_files = list(self.output_dir.glob("*_episode_*.parquet"))
        if not existing_files:
            return 1

        # Extract episode numbers
        episode_numbers = []
        for f in existing_files:
            try:
                parts = f.stem.split("_episode_")
                if len(parts) == 2:
                    episode_numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                continue

        return max(episode_numbers) + 1 if episode_numbers else 1

    def start_episode(self, custom_metadata: Optional[Dict[str, Any]] = None):
        """Start a new episode.
        
        Args:
            custom_metadata: Optional custom metadata to add to the episode
        """
        if self.current_episode is not None:
            print(f"Warning: Previous episode not saved. Saving it now.")
            self.save_episode()

        self.episode_start_time = datetime.now()

        # Create episode metadata
        metadata = EpisodeMetadata(robot_name=self.robot.name,
                                   episode_id=f"episode_{self.episode_counter:03d}",
                                   start_time=self.episode_start_time,
                                   duration=0.0,
                                   num_frames=0,
                                   recording_config={
                                       'record_velocities': self.record_velocities,
                                       'record_accelerations': self.record_accelerations,
                                       'contact_threshold': self.contact_threshold,
                                   })

        # Add URDF path if provided
        if self.urdf_path:
            metadata.add_custom_field('urdf_path', self.urdf_path)

        # Add custom metadata if provided
        if custom_metadata:
            for key, value in custom_metadata.items():
                metadata.add_custom_field(key, value)

        self.current_episode = EpisodeData(metadata=metadata, frames=[])
        self.prev_frame_data = None
        self.prev_timestamp = None

    def record_frame(
        self,
        joint_cfg: np.ndarray,
        base_pose: Optional[Union[jaxlie.SE3, Tuple[np.ndarray, np.ndarray]]] = None,
        timestamp: Optional[float] = None,
    ):
        """Record a single frame of data.
        
        Args:
            joint_cfg: Joint configuration array
            base_pose: Optional base pose as SE3 or (position, wxyz) tuple
            timestamp: Optional timestamp (seconds from episode start).
                      If None, uses time since episode start.
        """
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        # Compute timestamp if not provided
        if timestamp is None:
            timestamp = (datetime.now() - self.episode_start_time).total_seconds()

        # Convert inputs to numpy
        joint_cfg = np.array(joint_cfg)

        # Handle base pose
        if base_pose is None:
            # Fixed base - identity transform
            base_position = np.zeros(3)
            base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        elif isinstance(base_pose, jaxlie.SE3):
            base_position = np.array(base_pose.translation())
            base_wxyz = np.array(base_pose.rotation().wxyz)
        else:
            # Assume (position, wxyz) tuple
            base_position, base_wxyz = base_pose
            base_position = np.array(base_position)
            base_wxyz = np.array(base_wxyz)

        # Compute forward kinematics
        link_poses = self.robot.forward_kinematics(jnp.array(joint_cfg))

        # Transform to world frame if mobile base
        if base_pose is not None:
            base_SE3 = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(base_wxyz)),
                                                                jnp.array(base_position))
            link_poses_SE3 = jaxlie.SE3(link_poses)
            world_link_poses_SE3 = jax.vmap(lambda T: base_SE3 @ T)(link_poses_SE3)
            world_link_poses = world_link_poses_SE3.parameters()
        else:
            world_link_poses = link_poses

        # Extract link positions and rotations
        link_positions = np.array(jaxlie.SE3(world_link_poses).translation())
        link_rotations = np.array(jaxlie.SE3(world_link_poses).rotation().wxyz)

        # Compute velocities and accelerations
        dt = timestamp - self.prev_timestamp if self.prev_timestamp is not None else 0.0

        if self.record_velocities and self.prev_frame_data is not None and dt > 0:
            # Joint velocities
            joint_velocities = self.compute_functions.compute_joint_velocity(
                self.robot,
                joint_cfg,
                prev_positions=self.prev_frame_data['joint_positions'],
                dt=dt)

            # Link velocities
            link_linear_velocities = self.compute_functions.compute_link_velocity(
                self.robot,
                link_positions,
                prev_positions=self.prev_frame_data['link_positions'],
                dt=dt)

            link_angular_velocities = self.compute_functions.compute_link_angular_velocity(
                self.robot,
                link_positions,    # Pass positions for correct shape (M, 3)
                prev_positions=self.prev_frame_data['link_positions'],
                prev_rotations=self.prev_frame_data['link_rotations'],
                dt=dt)

            # Root velocities
            root_linear_velocity = self.compute_functions.compute_root_velocity(
                position=base_position, prev_position=self.prev_frame_data['root_position'], dt=dt)

            root_angular_velocity = self.compute_functions.compute_root_angular_velocity(
                rotation=base_wxyz, prev_rotation=self.prev_frame_data['root_quaternion'], dt=dt)
        else:
            # No previous frame or velocities disabled - set to zero
            joint_velocities = np.zeros_like(joint_cfg)
            link_linear_velocities = np.zeros_like(link_positions)
            link_angular_velocities = np.zeros_like(link_positions)
            root_linear_velocity = np.zeros(3)
            root_angular_velocity = np.zeros(3)

        # Compute accelerations
        if self.record_accelerations and self.prev_frame_data is not None and dt > 0:
            joint_accelerations = self.compute_functions.compute_joint_acceleration(
                self.robot,
                joint_velocities,
                prev_velocities=self.prev_frame_data.get('joint_velocities',
                                                         np.zeros_like(joint_cfg)),
                dt=dt)
        else:
            joint_accelerations = np.zeros_like(joint_cfg)

        # Detect contacts
        contact_status = self.compute_functions.detect_contact(self.robot,
                                                               link_positions,
                                                               list(self.robot.links.names),
                                                               threshold=self.contact_threshold)

        # Create frame data
        frame = FrameData(
            timestamp=timestamp,
            joint_names=list(self.robot.joints.actuated_names),
            joint_positions=joint_cfg,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            link_names=list(self.robot.links.names),
            link_positions=link_positions,
            link_rotations=link_rotations,
            link_linear_velocities=link_linear_velocities,
            link_angular_velocities=link_angular_velocities,
            root_position=base_position,
            root_quaternion=base_wxyz,
            root_linear_velocity=root_linear_velocity,
            root_angular_velocity=root_angular_velocity,
            contact_status=contact_status,
        )

        # Add frame to episode
        self.current_episode.add_frame(frame)

        # Store current frame data for next iteration
        self.prev_frame_data = {
            'joint_positions': joint_cfg,
            'joint_velocities': joint_velocities,
            'link_positions': link_positions,
            'link_rotations': link_rotations,
            'root_position': base_position,
            'root_quaternion': base_wxyz,
        }
        self.prev_timestamp = timestamp

    def save_episode(self, filename: Optional[str] = None) -> Path:
        """Save the current episode to a parquet file.
        
        Args:
            filename: Optional custom filename. If None, uses default format.
        
        Returns:
            Path to the saved file
        """
        if self.current_episode is None:
            raise RuntimeError("No episode to save.")

        if len(self.current_episode) == 0:
            print("Warning: Episode has no frames. Not saving.")
            return None

        # Generate filename if not provided
        if filename is None:
            date_str = self.episode_start_time.strftime("%Y-%m-%d")
            date_dir = self.output_dir / date_str
            date_dir.mkdir(exist_ok=True)

            filename = f"{self.robot.name}_episode_{self.episode_counter:03d}.parquet"
            filepath = date_dir / filename
        else:
            filepath = self.output_dir / filename

        # Convert episode to dataframe
        episode_dict = self.current_episode.to_dict()

        # Create table from episode data
        # We need to handle the nested structure properly
        flat_data = {}

        # Add metadata as table metadata
        metadata = episode_dict.pop('metadata')
        num_joints = episode_dict.pop('num_joints')
        num_links = episode_dict.pop('num_links')

        # Flatten the data for parquet
        for key, values in episode_dict.items():
            if key in ['joint_names', 'link_names']:
                # These are the same for all frames, store as metadata
                continue
            flat_data[key] = values

        # Create pandas DataFrame
        df = pd.DataFrame(flat_data)

        # Create pyarrow table with metadata
        table = pa.Table.from_pandas(df)

        # Add metadata to schema
        meta_dict = {
            'metadata': str(metadata),
            'num_joints': str(num_joints),
            'num_links': str(num_links),
            'joint_names': str(self.current_episode.frames[0].joint_names),
            'link_names': str(self.current_episode.frames[0].link_names),
        }

        table = table.replace_schema_metadata(meta_dict)

        # Write to parquet
        pq.write_table(table, filepath)

        print(f"Episode saved to: {filepath}")

        # Reset for next episode
        self.current_episode = None
        self.episode_counter += 1
        self.prev_frame_data = None
        self.prev_timestamp = None

        return filepath

    @staticmethod
    def load_episode(filepath: Union[str, Path]) -> EpisodeData:
        """Load an episode from a parquet file.
        
        Args:
            filepath: Path to the parquet file
        
        Returns:
            Loaded episode data
        """
        # Read parquet file
        table = pq.read_table(filepath)

        # Extract metadata
        meta_dict = table.schema.metadata
        if meta_dict is None:
            raise ValueError("No metadata found in parquet file")

        # Convert bytes to strings and parse
        import ast
        metadata = ast.literal_eval(meta_dict[b'metadata'].decode())
        num_joints = int(meta_dict[b'num_joints'].decode())
        num_links = int(meta_dict[b'num_links'].decode())
        joint_names = ast.literal_eval(meta_dict[b'joint_names'].decode())
        link_names = ast.literal_eval(meta_dict[b'link_names'].decode())

        # Convert to pandas for easier handling
        df = table.to_pandas()

        # Reconstruct episode data
        episode_dict = {
            'metadata': metadata,
            'num_joints': num_joints,
            'num_links': num_links,
        }

        # Add frame data
        for col in df.columns:
            episode_dict[col] = df[col].tolist()

        # Add names (same for all frames)
        num_frames = len(df)
        episode_dict['joint_names'] = [joint_names] * num_frames
        episode_dict['link_names'] = [link_names] * num_frames

        return EpisodeData.from_dict(episode_dict)
