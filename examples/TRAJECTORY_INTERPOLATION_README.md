# Trajectory Interpolation with IK Recording

The `18_trajectory_interpolation_recording.py` example demonstrates how to:
1. Interpolate between initial and final target poses
2. Solve IK at each interpolated frame
3. Record all frames into a single episode using the data logger

## Usage

```bash
python 18_trajectory_interpolation_recording.py [options]
```

## Command Line Arguments

### Robot Loading
- `--robot-name`, `-r`: Robot name for loading from robot_descriptions (default: "g1")
- `--urdf`, `-u`: Path to URDF file (overrides robot-name if provided)

### Interpolation Parameters
- `--initial-base-pos`: Initial base position as x y z (default: 0.0 0.0 0.75)
- `--final-base-pos`: Final base position as x y z (default: 0.0 0.0 0.4)
- `--step-size`: Step size for interpolation in meters (default: 0.05)

### Output
- `--output-dir`: Output directory for recorded data (default: "logs/trajectory_interpolation")

## Examples

### Basic Usage
```bash
# Use default settings (G1 robot, interpolate base from z=0.75 to z=0.4)
python 18_trajectory_interpolation_recording.py
```

### Custom Base Positions
```bash
# Interpolate base from (0, 0, 0.8) to (0.2, 0, 0.5)
python 18_trajectory_interpolation_recording.py \
    --initial-base-pos 0 0 0.8 \
    --final-base-pos 0.2 0 0.5 \
    --step-size 0.02
```

### Using Custom URDF
```bash
# Load custom URDF and interpolate
python 18_trajectory_interpolation_recording.py \
    --urdf models/unitree/g1/g1_29dof_rev_1_0.urdf \
    --initial-base-pos 0 0 0.9 \
    --final-base-pos 0 0 0.3
```

## How It Works

1. **Initialization**: 
   - Loads the robot URDF
   - Sets up the visualizer and data logger
   - Calculates the number of interpolation steps based on distance and step size

2. **Interpolation**:
   - Linearly interpolates between initial and final base positions
   - Maintains relative positions for other target links (ankles, palms)
   - Currently only interpolates positions (rotation interpolation to be added later)

3. **IK Solving**:
   - For each interpolated frame:
     - Updates target positions
     - Solves IK using `solve_ik_with_multiple_targets_and_base`
     - Uses high weights for feet to keep them stable
     - Enables COM support polygon constraint

4. **Recording**:
   - Click "Start Recording" button to begin
   - Records each frame with the data logger
   - Saves all frames as a single episode when complete
   - Includes metadata about interpolation parameters

## Output

The recorded data is saved as a parquet file in the specified output directory with:
- Robot configurations at each frame
- Base poses at each frame
- Interpolation metadata
- Timing information

## Features

- **Modular Design**: Easy to extend for rotation interpolation
- **Real-time Visualization**: See the robot motion during recording
- **Progress Tracking**: Visual progress bar and frame counter
- **COM Support**: Ensures stable poses with COM support polygon constraint

## Notes

- The example currently only interpolates positions, not rotations
- Other target links (ankles, palms) maintain their relative positions to the base
- The IK solver uses high weights for feet to maintain ground contact
- Each frame is solved independently (no trajectory optimization)