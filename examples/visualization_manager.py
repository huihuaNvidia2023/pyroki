"""Visualization management for trajectory playback."""

from typing import Dict, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jaxlie
import viser
from viser.extras import ViserUrdf
import pyroki as pk


class VisualizationManager:
    """Manages trajectory visualization and playback."""
    
    def __init__(self, server: viser.ViserServer, robot: pk.Robot, urdf):
        """
        Initialize visualization manager.
        
        Args:
            server: Viser server instance
            robot: PyRoKi robot instance
            urdf: URDF model
        """
        self.server = server
        self.robot = robot
        self.urdf = urdf
        
        # Robot visualization
        self.base_frame = server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
        
        # Support polygon visualization
        self.support_viz = pk.viewer.SupportPolygonVisualizer(
            server,
            robot,
            root_node_name="/support_polygon",
            com_color=(255, 50, 50),
            polygon_color=(50, 255, 50),
            com_radius=0.03,
            show_com_line=True,
            com_line_height=1.0,
            visible=True
        )
        
        # Pose markers for start/end visualization
        self.pose_markers: Dict[str, viser.SceneNodeHandle] = {}
        
        # Track label visibility
        self.labels_visible = True
        
    def show_pose_markers(self, start_positions: Dict[str, np.ndarray],
                         start_wxyzs: Dict[str, np.ndarray],
                         end_positions: Dict[str, np.ndarray] = None,
                         end_wxyzs: Dict[str, np.ndarray] = None) -> None:
        """Show visual markers for captured start/end poses."""
        # Clear existing pose markers
        self._clear_pose_markers()
        
        # Show start pose markers
        for link_name, position in start_positions.items():
            wxyz = start_wxyzs.get(link_name, np.array([1.0, 0.0, 0.0, 0.0]))
            
            # Create start marker
            start_frame = self.server.scene.add_frame(
                f"/pose_marker_start_{link_name}",
                position=position,
                wxyz=wxyz,
                axes_length=0.05,
                axes_radius=0.01
            )
            self.pose_markers[f"start_{link_name}"] = start_frame
            
            # Add label
            label = self.server.scene.add_label(
                f"/pose_marker_start_{link_name}/label",
                text=f"Start: {link_name.replace('_', ' ').title()}",
                position=(0, 0, 0.05)
            )
            label.visible = self.labels_visible
            self.pose_markers[f"start_{link_name}_label"] = label
            
        # Show end pose markers if provided
        if end_positions is not None:
            for link_name, position in end_positions.items():
                wxyz = end_wxyzs.get(link_name, np.array([1.0, 0.0, 0.0, 0.0]))
                
                # Create end marker with different color scheme
                end_frame = self.server.scene.add_frame(
                    f"/pose_marker_end_{link_name}",
                    position=position,
                    wxyz=wxyz,
                    axes_length=0.05,
                    axes_radius=0.01
                )
                self.pose_markers[f"end_{link_name}"] = end_frame
                
                # Add label
                label = self.server.scene.add_label(
                    f"/pose_marker_end_{link_name}/label",
                    text=f"End: {link_name.replace('_', ' ').title()}",
                    position=(0, 0, 0.05)
                )
                label.visible = self.labels_visible
                self.pose_markers[f"end_{link_name}_label"] = label
                
    def _clear_pose_markers(self) -> None:
        """Clear all pose markers."""
        for marker in self.pose_markers.values():
            marker.remove()
        self.pose_markers.clear()
        
    def update_robot_pose(self, base_position: np.ndarray,
                         base_wxyz: np.ndarray,
                         joint_cfg: np.ndarray) -> None:
        """Update robot visualization with current pose."""
        self.base_frame.position = base_position
        self.base_frame.wxyz = base_wxyz
        
        # Convert to numpy if it's a JAX array
        if hasattr(joint_cfg, '__array__'):
            joint_cfg_np = np.array(joint_cfg)
        else:
            joint_cfg_np = joint_cfg
            
        self.urdf_vis.update_cfg(joint_cfg_np)
        
        # Update support polygon
        base_pose_SE3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(
                jnp.array([base_wxyz[1], base_wxyz[2], base_wxyz[3], base_wxyz[0]])),
            jnp.array(base_position)
        )
        self.support_viz.update(jnp.array(joint_cfg), base_pose_SE3)
        
    def get_com_status(self) -> str:
        """Get current COM status text."""
        return self.support_viz.get_status_text()
        
    def set_labels_visible(self, visible: bool) -> None:
        """Set visibility of all label markers.
        
        Args:
            visible: Whether labels should be visible
        """
        self.labels_visible = visible
        
        # Update visibility of existing labels
        for key, marker in self.pose_markers.items():
            if key.endswith('_label'):
                marker.visible = visible
                
    def toggle_labels(self) -> None:
        """Toggle visibility of all label markers."""
        self.set_labels_visible(not self.labels_visible)
        
    def set_support_polygon_visible(self, visible: bool) -> None:
        """Show or hide support polygon visualization."""
        self.support_viz.set_visibility(visible)
        
    def reset_support_polygon(self) -> None:
        """Reset the support polygon visualizer to avoid stale handle issues."""
        # Get current visibility state before reset
        current_visible = getattr(self.support_viz, '_visible', True)
        
        # Clear any existing handles
        try:
            self.support_viz._clear_handles()
        except:
            pass  # Ignore errors if handles already removed
            
        # Recreate the support polygon visualizer
        self.support_viz = pk.viewer.SupportPolygonVisualizer(
            self.server,
            self.robot,
            root_node_name="/support_polygon",
            com_color=(255, 50, 50),
            polygon_color=(50, 255, 50),
            com_radius=0.03,
            show_com_line=True,
            com_line_height=1.0,
            visible=current_visible  # Preserve visibility state
        ) 