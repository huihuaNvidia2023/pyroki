# Loading URDF Files in Pyroki Examples

The modified `13_mobile_humanoid_ik_with_logging.py` example now supports loading URDF files from multiple sources using two separate arguments:

## Usage

```bash
python 13_mobile_humanoid_ik_with_logging.py [--robot-name <name>] [--urdf <path>]
```

## Arguments

- `--robot-name`: Robot name for loading from `robot_descriptions` package (default: "g1")
- `--urdf`: Path to URDF file (if provided, overrides `--robot-name`)

## Examples

### 1. Load from robot_descriptions (Default)
```bash
# Default (loads g1_description)
python 13_mobile_humanoid_ik_with_logging.py

# Specify different robot name
python 13_mobile_humanoid_ik_with_logging.py --robot-name panda
python 13_mobile_humanoid_ik_with_logging.py --robot-name ur5
```

### 2. Load from URDF file
```bash
# Load from relative path
python 13_mobile_humanoid_ik_with_logging.py --urdf models/unitree/g1/g1_29dof_rev_1_0.urdf

# Load from absolute path
python 13_mobile_humanoid_ik_with_logging.py --urdf /home/user/my_robot.urdf

# URDF file takes precedence over robot-name
python 13_mobile_humanoid_ik_with_logging.py --robot-name panda --urdf custom_robot.urdf
```

## How It Works

The script prioritizes the `--urdf` argument:
- If `--urdf` is provided (non-empty), it loads the URDF from the specified file path
- If `--urdf` is not provided or empty, it uses `--robot-name` to load from `robot_descriptions`

When loading from a file:
- The script uses `yourdfpy.URDF.load()` to parse the URDF
- A custom `filename_handler` is provided to resolve mesh paths relative to the URDF file location
- The loaded URDF is then passed to `pk.Robot.from_urdf()` as usual

## Implementation Details

The key changes are:

1. **Import required modules:**
```python
import argparse
from pathlib import Path
import yourdfpy
```

2. **Add argument parser with two separate arguments:**
```python
parser = argparse.ArgumentParser(description="Mobile Humanoid IK with Data Logging")
parser.add_argument(
    "--robot-name",
    type=str,
    help="Robot name for loading from robot_descriptions (e.g., 'g1', 'panda', 'ur5')",
    default="g1"
)
parser.add_argument(
    "--urdf",
    type=str,
    help="Path to URDF file (overrides robot-name if provided)",
    default=""
)
args = parser.parse_args()
```

3. **Load URDF based on arguments:**
```python
if args.urdf:
    # Load from URDF file path
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = Path.cwd() / urdf_path
    
    def filename_handler(fname: str) -> str:
        base_path = urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    urdf = yourdfpy.URDF.load(str(urdf_path), filename_handler=filename_handler)
else:
    # Load from robot_descriptions using robot name
    urdf = load_robot_description(f"{args.robot_name}_description")
```

## Notes

- The `filename_handler` is important for resolving mesh file paths referenced in the URDF
- Make sure mesh files are in the correct location relative to the URDF file
- The `--urdf` argument takes precedence over `--robot-name` if both are provided
- The rest of the example code remains unchanged - it works with any loaded URDF