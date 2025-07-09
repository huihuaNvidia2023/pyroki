"""Interactive marker management for trajectory planning."""

from typing import Dict, List, Tuple
import numpy as np
import viser


class InteractiveMarkerManager:
    """Manages interactive transform controls for multiple links."""
    
    def __init__(self, server: viser.ViserServer, link_names: List[str],
                 initial_positions: Dict[str, np.ndarray] = None,
                 initial_wxyzs: Dict[str, np.ndarray] = None):
        """
        Initialize marker manager.
        
        Args:
            server: Viser server instance
            link_names: List of link names to create markers for
            initial_positions: Initial positions for each link
            initial_wxyzs: Initial orientations for each link
        """
        self.server = server
        self.link_names = link_names
        self.markers: Dict[str, viser.TransformControlsHandle] = {}
        
        # Create markers with initial poses
        self._create_markers(initial_positions, initial_wxyzs)
        
    def _create_markers(self, initial_positions: Dict[str, np.ndarray] = None,
                       initial_wxyzs: Dict[str, np.ndarray] = None) -> None:
        """Create transform control markers for each link."""
        for link_name in self.link_names:
            # Determine initial pose
            if initial_positions and link_name in initial_positions:
                position = initial_positions[link_name]
            else:
                # Default positions based on link type
                if "ankle" in link_name:
                    position = np.array([0.05, 0.1 if "left" in link_name else -0.1, 0.05])
                elif "palm" in link_name:
                    position = np.array([0.41, 0.2 if "left" in link_name else -0.2, 0.85])
                else:
                    position = np.array([0.0, 0.0, 0.5])
                    
            if initial_wxyzs and link_name in initial_wxyzs:
                wxyz = initial_wxyzs[link_name]
            else:
                wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Create marker
            marker = self.server.scene.add_transform_controls(
                f"/marker_{link_name}",
                scale=0.2,
                position=position,
                wxyz=wxyz
            )
            self.markers[link_name] = marker
            
            # Add label
            self.server.scene.add_label(
                f"/marker_{link_name}/label",
                text=link_name.replace("_", " ").title(),
                position=(0, 0, 0.1)
            )
    
    def get_poses(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get current poses of all markers.
        
        Returns:
            Tuple of (positions_dict, wxyzs_dict)
        """
        positions = {}
        wxyzs = {}
        
        for name, marker in self.markers.items():
            positions[name] = np.array(marker.position)
            wxyzs[name] = np.array(marker.wxyz)
            
        return positions, wxyzs
    
    def set_poses(self, positions: Dict[str, np.ndarray],
                  wxyzs: Dict[str, np.ndarray]) -> None:
        """Set marker poses from saved data."""
        for name, marker in self.markers.items():
            if name in positions:
                marker.position = positions[name]
            if name in wxyzs:
                marker.wxyz = wxyzs[name]
                
    def set_visible(self, visible: bool) -> None:
        """Show or hide all markers."""
        for marker in self.markers.values():
            marker.visible = visible
            
    def highlight_markers(self, highlight: bool, color: Tuple[int, int, int] = None) -> None:
        """Highlight markers to indicate they're being captured."""
        # Note: Viser doesn't directly support color changes for transform controls
        # This is a placeholder for potential future functionality
        pass 