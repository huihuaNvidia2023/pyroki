"""Support polygon and COM visualization for robots."""

from typing import Optional, Tuple, List, Union

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import viser
from jax.typing import ArrayLike
from loguru import logger

from .._robot import Robot
from ..robots_config import compute_foot_local_corners, get_foot_link_names


class SupportPolygonVisualizer:
    """Helper class to visualize the support polygon and center of mass projection for a robot."""

    def __init__(
        self,
        server: Optional[Union[viser.ViserServer, viser.ClientHandle]],
        robot: Robot,
        robot_description: str,
        root_node_name: str = "/support_polygon",
        com_color: Tuple[int, int, int] = (255, 0, 0),
        polygon_color: Tuple[int, int, int] = (0, 255, 0),
        com_radius: float = 0.02,
        corner_radius: float = 0.01,
        edge_radius: float = 0.003,
        visible: bool = True,
        show_com_line: bool = True,
        com_line_height: float = 0.5,
    ):
        """Initializes the support polygon visualizer.

        Args:
            server: The Viser server or client handle. Can be None for calculation-only usage.
            robot: The Pyroki robot model.
            robot_description: The robot description name (e.g., "g1_description").
            root_node_name: The base name for visualization elements in the Viser scene.
            com_color: The color of the COM projection.
            polygon_color: The color of the support polygon.
            com_radius: Radius of the COM sphere.
            corner_radius: Radius of the polygon corner spheres.
            edge_radius: Radius of the edge line spheres.
            visible: Initial visibility state.
            show_com_line: Whether to show a vertical line from COM to ground.
            com_line_height: Height of the COM line visualization.
        """
        self._server = server
        self._robot = robot
        self._robot_description = robot_description
        self._root_node_name = root_node_name
        self._com_color = com_color
        self._polygon_color = polygon_color
        self._com_radius = com_radius
        self._corner_radius = corner_radius
        self._edge_radius = edge_radius
        self._visible = visible
        self._show_com_line = show_com_line
        self._com_line_height = com_line_height

        # Get foot link information
        self._foot_link_names = get_foot_link_names(robot_description)
        self._foot_link_indices = jnp.array(
            [robot.links.names.index(name) for name in self._foot_link_names])

        # Precompute local foot corners
        self._local_corners = compute_foot_local_corners(robot_description=robot_description)

        # Visualization handles
        self._com_handle: Optional[viser.SceneNodeHandle] = None
        self._com_line_handles: List[viser.SceneNodeHandle] = []
        self._corner_handles: List[viser.SceneNodeHandle] = []
        self._edge_handles: List[viser.SceneNodeHandle] = []

        # State tracking
        self._last_joints: Optional[jnp.ndarray] = None
        self._last_base_pose: Optional[jaxlie.SE3] = None
        self.com_xy: Optional[jnp.ndarray] = None
        self.corners_xy: Optional[jnp.ndarray] = None
        self.is_inside: bool = False
        self.min_margin: float = 0.0

    def _clear_handles(self):
        """Removes all visualization handles from the scene."""
        if self._com_handle is not None:
            self._com_handle.remove()
            self._com_handle = None

        for handle in self._com_line_handles:
            handle.remove()
        self._com_line_handles = []

        for handle in self._corner_handles:
            handle.remove()
        self._corner_handles = []

        for handle in self._edge_handles:
            handle.remove()
        self._edge_handles = []

    def update(self, joints: ArrayLike, base_pose: Optional[jaxlie.SE3] = None):
        """Updates the visualization based on the current configuration.

        Args:
            joints: The current joint angles of the robot.
            base_pose: Optional base pose for mobile robots.
        """
        joints = jnp.asarray(joints)
        self._last_joints = joints
        self._last_base_pose = base_pose

        try:
            # Always compute COM and support polygon
            self._compute_com_and_polygon(joints, base_pose)

            # Only update visualization if server is available and visible
            if self._server is not None and self._visible:
                self._update_visualization()

        except Exception as e:
            logger.warning(f"Failed to update support polygon: {e}")
            if self._server is not None:
                self._clear_handles()

    def _compute_com_and_polygon(self,
                                 joint_config: jnp.ndarray,
                                 base_pose: Optional[jaxlie.SE3] = None):
        """Computes the COM position and support polygon vertices."""
        # Get link poses
        if base_pose is None:
            link_poses = self._robot.forward_kinematics(joint_config)
            link_poses_SE3 = jaxlie.SE3(link_poses)
        else:
            # Transform to world frame for mobile base case
            link_poses_base = self._robot.forward_kinematics(joint_config)
            link_poses_base_SE3 = jaxlie.SE3(link_poses_base)
            link_poses_SE3 = jax.vmap(lambda T: base_pose @ T)(link_poses_base_SE3)

        # Compute COM
        all_link_positions = link_poses_SE3.translation()
        masses = self._robot.links.masses[:, None]
        total_mass = masses.sum()

        com_position = jnp.where(total_mass > 1e-6,
                                 (all_link_positions * masses).sum(axis=0) / total_mass,
                                 all_link_positions.mean(axis=0))
        self.com_xy = jnp.array([com_position[0], com_position[1]])

        # Get foot poses and compute support polygon vertices
        if base_pose is None:
            # Direct indexing for basic case
            foot_poses_wxyz_xyz = link_poses[self._foot_link_indices]
            foot_poses = jaxlie.SE3(foot_poses_wxyz_xyz)
        else:
            # Extract parameters and index for mobile base case
            link_poses_params = link_poses_SE3.parameters()
            foot_poses_params = link_poses_params[self._foot_link_indices]
            foot_poses = jaxlie.SE3(foot_poses_params)

        # Transform corners to world
        def transform_foot_corners(foot_pose):
            world_corners = jax.vmap(foot_pose.apply)(self._local_corners)
            return world_corners

        world_corners = jax.vmap(transform_foot_corners)(foot_poses)    # [F, 4, 3]
        self.corners_xy = world_corners[:, :, [0, 1]].reshape(-1, 2)    # [F*4, 2]

        # Check if COM is inside polygon
        self._check_com_in_polygon()

    def _check_com_in_polygon(self):
        """Checks if the COM is inside the support polygon and computes margins."""
        # Sample directions and check margins
        num_dirs = 16
        angles = jnp.linspace(0, 2 * jnp.pi, num_dirs, endpoint=False)
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        corners_proj = self.corners_xy @ directions.T
        com_proj = self.com_xy @ directions.T
        hull_support = corners_proj.max(axis=0)
        margins = hull_support - com_proj

        # COM is inside if all margins are positive
        self.is_inside = bool(jnp.all(margins >= -1e-6))    # Small tolerance for numerical errors
        self.min_margin = float(margins.min())

    def _update_visualization(self):
        """Updates the visualization elements in the scene."""
        # Clear existing handles
        self._clear_handles()

        if self.com_xy is None or self.corners_xy is None or self._server is None:
            return

        # Visualize COM projection
        self._com_handle = self._server.scene.add_icosphere(f"{self._root_node_name}/com",
                                                            radius=self._com_radius,
                                                            color=self._com_color,
                                                            position=(float(self.com_xy[0]),
                                                                      float(self.com_xy[1]), 0.02))

        # Add vertical line from COM to ground if enabled
        if self._show_com_line:
            num_points = 20
            for i in range(num_points):
                z = self._com_line_height * (1 - i / (num_points - 1))
                handle = self._server.scene.add_icosphere(
                    f"{self._root_node_name}/com_line/{i}",
                    radius=self._edge_radius * 0.7,
                    color=tuple(int(c * 0.8) for c in self._com_color),
                    position=(float(self.com_xy[0]), float(self.com_xy[1]), z))
                self._com_line_handles.append(handle)

        # Visualize polygon corners
        for i, corner in enumerate(self.corners_xy):
            handle = self._server.scene.add_icosphere(f"{self._root_node_name}/corner_{i}",
                                                      radius=self._corner_radius,
                                                      color=self._polygon_color,
                                                      position=(float(corner[0]), float(corner[1]),
                                                                0.01))
            self._corner_handles.append(handle)

        # Connect polygon corners with edges
        num_corners = len(self.corners_xy)
        if num_corners == 8:    # 2 feet with 4 corners each
            # Draw rectangles for each foot
            for foot_idx in range(2):
                foot_corners = self.corners_xy[foot_idx * 4:(foot_idx + 1) * 4]
                # Connect corners in order: 0-1-2-3-0
                for i in range(4):
                    start = foot_corners[i]
                    end = foot_corners[(i + 1) % 4]
                    self._draw_edge(start, end, f"foot{foot_idx}_edge_{i}")
        else:
            # For other cases, just connect consecutive corners
            for i in range(num_corners):
                start = self.corners_xy[i]
                end = self.corners_xy[(i + 1) % num_corners]
                self._draw_edge(start, end, f"edge_{i}")

    def _draw_edge(self, start: jnp.ndarray, end: jnp.ndarray, edge_name: str):
        """Draws an edge between two points using small spheres."""
        if self._server is None:
            return

        num_points = 10
        for j in range(num_points):
            t = j / (num_points - 1)
            point = start * (1 - t) + end * t
            handle = self._server.scene.add_icosphere(
                f"{self._root_node_name}/{edge_name}_{j}",
                radius=self._edge_radius,
                color=tuple(int(c * 0.8) for c in self._polygon_color),
                position=(float(point[0]), float(point[1]), 0.01))
            self._edge_handles.append(handle)

    def set_visibility(self, visible: bool):
        """Sets the visibility of the support polygon visualization."""
        self._visible = visible
        if not visible and self._server is not None:
            self._clear_handles()
        elif visible and self._last_joints is not None:
            # Re-update with last known configuration
            self.update(self._last_joints, self._last_base_pose)

    def get_status_text(self) -> str:
        """Returns a status string describing the current COM/polygon state."""
        if self.com_xy is None:
            return "No data"

        status = "✓ Inside" if self.is_inside else "✗ Outside"
        return f"{status} | COM: ({self.com_xy[0]:.3f}, {self.com_xy[1]:.3f}) | Margin: {self.min_margin:.3f}m"

    def remove(self):
        """Removes all visualization elements from the scene."""
        if self._server is not None:
            self._clear_handles()
        self._last_joints = None
        self._last_base_pose = None
        self.com_xy = None
        self.corners_xy = None
