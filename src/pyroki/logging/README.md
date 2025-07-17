# PyRoKi Data Logging Module

The PyRoKi logging module provides comprehensive data logging functionality for robot trajectories and states, with support for both single frames and full trajectories.

## Features

- **Comprehensive Data Logging**: Records joint positions, velocities, accelerations, link poses, velocities, and contact states
- **Mobile Base Support**: Handles both fixed-base and mobile robots
- **Parquet File Format**: Efficient columnar storage with metadata support
- **Extensible Metadata**: Add custom fields for experiment tracking
- **Pluggable Compute Functions**: Customize velocity/acceleration computation and contact detection
- **Episode Management**: Automatic episode numbering and organization

## Quick Start

```python
import pyroki as pk
import numpy as np

# Load robot
robot = pk.Robot.from_urdf(urdf)

# Create logger
logger = pk.logging.DataLogger(
    robot=robot,
    output_dir="logs",
    contact_threshold=0.05,
    record_velocities=True,
    record_accelerations=True,
)

# Start episode with metadata
logger.start_episode(custom_metadata={
    'experiment': 'reaching_task',
    'trial': 1,
})

# Record frames
for t in range(timesteps):
    logger.record_frame(
        joint_cfg=joint_positions[t],
        base_pose=base_poses[t],  # Optional for mobile base
        timestamp=t * dt
    )

# Save episode
filepath = logger.save_episode()

# Load episode
loaded_episode = pk.logging.DataLogger.load_episode(filepath)
```

## Data Structure

Each episode contains:

### Frame Data
- `timestamp`: Time relative to episode start (seconds)
- `joint_names`: List of actuated joint names
- `joint_positions`: Joint configuration array
- `joint_velocities`: Joint velocities (computed or user-provided)
- `joint_accelerations`: Joint accelerations (computed or user-provided)
- `link_names`: List of all link names
- `link_positions`: World positions of all links (M, 3)
- `link_rotations`: World rotations as quaternions (M, 4) - wxyz format
- `link_linear_velocities`: Linear velocities of links (M, 3)
- `link_angular_velocities`: Angular velocities of links (M, 3)
- `root_position`: Base/root position (3,)
- `root_quaternion`: Base/root orientation (4,) - wxyz format
- `root_linear_velocity`: Base linear velocity (3,)
- `root_angular_velocity`: Base angular velocity (3,)
- `contact_status`: Boolean array indicating contact for each link (M,)

### Episode Metadata
- `robot_name`: Name of the robot
- `episode_id`: Unique episode identifier
- `start_time`: Episode start timestamp
- `duration`: Total duration in seconds
- `num_frames`: Number of frames in episode
- `recording_config`: Recording configuration (velocities, contact threshold, etc.)
- `custom_metadata`: User-defined metadata dictionary

## File Organization

Files are saved in the following structure:
```
logs/
└── YYYY-MM-DD/
    ├── robot_name_episode_001.parquet
    ├── robot_name_episode_002.parquet
    └── ...
```

## Custom Compute Functions

You can provide custom functions for computing velocities, accelerations, and contact detection:

```python
def custom_contact_detector(robot, link_positions, link_names, threshold=0.05, **kwargs):
    # Custom logic for detecting contacts
    contacts = link_positions[:, 2] < threshold
    return contacts

compute_fns = pk.logging.ComputeFunctions(
    detect_contact=custom_contact_detector,
    compute_joint_velocity=my_velocity_function,
)

logger = pk.logging.DataLogger(
    robot=robot,
    compute_functions=compute_fns,
)
```

## Integration Examples

### Single Frame Recording (Use Case 1)
See `examples/13_mobile_humanoid_ik_with_logging.py` for an example of recording single frames with a button click.

### Trajectory Recording (Use Case 2)
See `examples/14_mobile_humanoid_trajopt_with_logging.py` for an example of recording entire trajectories.

### Simple Demo
Run `examples/logging_demo.py` for a comprehensive demonstration of all features.

## Loading and Analysis

```python
# Load episode
episode = pk.logging.DataLogger.load_episode("path/to/episode.parquet")

# Access metadata
print(f"Robot: {episode.metadata.robot_name}")
print(f"Duration: {episode.metadata.duration}s")
print(f"Custom data: {episode.metadata.custom_metadata}")

# Access frame data
for frame in episode.frames:
    print(f"Time: {frame.timestamp}")
    print(f"Joint positions: {frame.joint_positions}")
    print(f"Contacts: {np.sum(frame.contact_status)} links in contact")
```

## Dependencies

- `pandas`: DataFrame operations
- `pyarrow`: Parquet file I/O
- `numpy`: Array operations
- `jax/jaxlie`: Pose transformations

## Tips

1. **Contact Detection**: Adjust `contact_threshold` based on your robot and environment
2. **Performance**: Disable velocity/acceleration recording if not needed to improve performance
3. **Storage**: Parquet files are compressed; typical episodes are 10-100KB
4. **Custom Data**: Use `custom_metadata` to track experiment parameters, user notes, etc.

## Playback Viewer

PyRoKi includes an interactive playback viewer for visualizing logged episodes:

```python
import pyroki as pk

# Play back a single episode
player = pk.viewer.EpisodePlayback("logs/2024-01-01/robot_episode_001.parquet")
player.run()

# Play back all episodes in a directory
player = pk.viewer.EpisodePlayback("logs/2024-01-01/")
player.run()
```

Features:
- Automatic robot loading from episode metadata
- Episode navigation with Previous/Next buttons
- Frame-by-frame playback with adjustable speed
- Support polygon and COM visualization
- Contact status display
- Custom metadata viewing

See `examples/17_episode_playback_demo.py` for more details. 