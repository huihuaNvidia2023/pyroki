"""Unit test for link mass parsing in RobotURDFParser."""

import os
import tempfile
import numpy as np
import yourdfpy
from pyroki._robot_urdf_parser import RobotURDFParser


def test_link_mass_parsing():
    """Test that link masses are correctly parsed from URDF."""
    
    # Create a minimal URDF with links having different mass properties
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <!-- Base link with mass -->
  <link name="base_link">
    <inertial>
      <mass value="10.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Link with inertial but no mass (should default to 0.0) -->
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  
  <!-- Link with no inertial properties (should default to 0.0) -->
  <link name="link2">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
  
  <!-- Link with mass -->
  <link name="link3">
    <inertial>
      <mass value="2.3"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  
  <!-- Joints to connect the links -->
  <joint name="joint1" type="fixed">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 1 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="10.0"/>
  </joint>
  
  <joint name="joint3" type="fixed">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>
"""
    
    # Write URDF to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        urdf_path = f.name
    
    try:
        # Load URDF using yourdfpy
        urdf = yourdfpy.URDF.load(urdf_path)
        
        # Parse using RobotURDFParser
        joint_info, link_info = RobotURDFParser.parse(urdf)
        
        # Verify link names
        expected_names = ('base_link', 'link1', 'link2', 'link3')
        assert link_info.names == expected_names, f"Expected names {expected_names}, got {link_info.names}"
        
        # Verify masses
        expected_masses = np.array([10.5, 0.0, 0.0, 2.3], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            link_info.masses, 
            expected_masses,
            decimal=5,
            err_msg="Link masses do not match expected values"
        )
        
        # Verify shapes
        assert link_info.masses.shape == (4,), f"Expected shape (4,), got {link_info.masses.shape}"
        assert link_info.num_links == 4, f"Expected 4 links, got {link_info.num_links}"
        
        print("✓ All tests passed!")
        print(f"Link names: {link_info.names}")
        print(f"Link masses: {link_info.masses}")
        
    finally:
        # Clean up temporary file
        os.unlink(urdf_path)


if __name__ == "__main__":
    test_link_mass_parsing() 