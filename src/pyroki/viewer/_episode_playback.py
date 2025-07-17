"""Episode playback viewer for logged robot data.

This module provides a self-contained viewer for playing back logged episodes
from PyRoKi's data logging module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import jax.numpy as jnp
import jaxlie
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

from .._robot import Robot
from ..logging import DataLogger, EpisodeData
from ._support_polygon import SupportPolygonVisualizer


class EpisodePlayback:
    """Viewer for playing back logged robot episodes.
    
    Example usage:
        # Play back a single episode
        player = EpisodePlayback("logs/2024-01-01/robot_episode_001.parquet")
        player.run()
        
        # Play back multiple episodes
        player = EpisodePlayback(["episode1.parquet", "episode2.parquet"])
        player.run()
        
        # Play back all episodes in a directory
        player = EpisodePlayback("logs/2024-01-01/")
        player.run()
    """

    def __init__(
        self,
        episodes: Union[str, Path, List[Union[str, Path]]],
        server: Optional[viser.ViserServer] = None,
        show_support_polygon: bool = True,
    ):
        """Initialize episode playback viewer.
        
        Args:
            episodes: Path to episode file(s) or directory containing episodes
            server: Optional existing viser server, creates new one if None
            show_support_polygon: Whether to show COM/support polygon visualization
        """
        # Create or use existing server
        self.server = server or viser.ViserServer()
        self.show_support_polygon = show_support_polygon

        # Load episodes
        self.episodes = self._load_episodes(episodes)
        if not self.episodes:
            raise ValueError("No episodes found to playback")

        # Initialize robot and visualizers (will be set when first episode loads)
        self.robot: Optional[Robot] = None
        self.urdf = None
        self.urdf_vis: Optional[ViserUrdf] = None
        self.support_viz: Optional[SupportPolygonVisualizer] = None

        # Playback state
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.playing = False
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        # Setup UI
        self._setup_ui()

        # Load first episode
        self._load_episode(0)

    def _load_episodes(self, episodes: Union[str, Path, List[Union[str,
                                                                   Path]]]) -> List[EpisodeData]:
        """Load episode data from files or directory."""
        episode_list = []

        # Convert to list of paths
        if isinstance(episodes, (str, Path)):
            path = Path(episodes)
            if path.is_dir():
                # Load all parquet files in directory (recursively)
                episode_files = sorted(path.rglob("*.parquet"))
            else:
                # Single file
                episode_files = [path]
        else:
            # List of files
            episode_files = [Path(ep) for ep in episodes]

        # Load each episode
        for filepath in episode_files:
            if filepath.exists() and filepath.suffix == ".parquet":
                try:
                    episode = DataLogger.load_episode(filepath)
                    episode_list.append(episode)
                    print(f"Loaded episode: {filepath.name}")
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")

        return episode_list

    def _setup_ui(self):
        """Setup viser GUI controls."""
        # Add grid
        self.server.scene.add_grid("/ground", width=5, height=5, cell_size=0.1)

        # Episode info
        with self.server.gui.add_folder("Episode Info"):
            self.episode_text = self.server.gui.add_text("Current Episode",
                                                         "No episode loaded",
                                                         disabled=True)
            self.robot_text = self.server.gui.add_text("Robot", "Unknown", disabled=True)
            self.duration_text = self.server.gui.add_text("Duration", "0.0s", disabled=True)
            self.metadata_text = self.server.gui.add_text("Metadata", "{}", disabled=True)

        # Episode controls
        with self.server.gui.add_folder("Episode Controls"):
            self.episode_slider = self.server.gui.add_slider(
                "Episode",
                min=0,
                max=max(0,
                        len(self.episodes) - 1),
                step=1,
                initial_value=0,
            )
            self.episode_slider.on_update(self._on_episode_change)

            self.prev_episode_btn = self.server.gui.add_button("Previous Episode")
            self.prev_episode_btn.on_click(self._prev_episode)

            self.next_episode_btn = self.server.gui.add_button("Next Episode")
            self.next_episode_btn.on_click(self._next_episode)

        # Frame controls
        with self.server.gui.add_folder("Frame Controls"):
            self.frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=0,
                step=1,
                initial_value=0,
            )
            self.frame_slider.on_update(self._on_frame_change)

            self.timestamp_text = self.server.gui.add_text("Timestamp", "0.00s", disabled=True)

            self.play_pause_btn = self.server.gui.add_button("Play")
            self.play_pause_btn.on_click(self._toggle_playback)

            self.speed_slider = self.server.gui.add_slider(
                "Playback Speed",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=1.0,
            )
            self.speed_slider.on_update(
                lambda _: setattr(self, 'playback_speed', self.speed_slider.value))

        # Visualization options
        with self.server.gui.add_folder("Visualization"):
            self.show_support_checkbox = self.server.gui.add_checkbox("Show Support Polygon",
                                                                      self.show_support_polygon)
            self.show_support_checkbox.on_update(self._toggle_support_viz)

            self.com_status_text = self.server.gui.add_text("COM Status", "N/A", disabled=True)

            self.contact_text = self.server.gui.add_text("Contacts", "None", disabled=True)

    def _load_episode(self, idx: int):
        """Load and setup visualization for an episode."""
        if idx < 0 or idx >= len(self.episodes):
            return

        self.current_episode_idx = idx
        episode = self.episodes[idx]

        # Update episode info
        self.episode_text.value = f"Episode {idx + 1} of {len(self.episodes)}"
        self.robot_text.value = episode.metadata.robot_name
        self.duration_text.value = f"{episode.metadata.duration:.2f}s"

        # Show custom metadata
        if episode.metadata.custom_metadata:
            import json
            self.metadata_text.value = json.dumps(episode.metadata.custom_metadata,
                                                  indent=2)[:200] + "..."
        else:
            self.metadata_text.value = "No custom metadata"

        # Load robot if needed (only on first episode or robot change)
        if self.robot is None or self.robot.name != episode.metadata.robot_name:
            self._setup_robot(episode.metadata.robot_name)

            # Update support polygon checkbox based on robot capabilities
            if self.support_viz is None:
                self.show_support_checkbox.value = False
                self.show_support_checkbox.disabled = True
            else:
                self.show_support_checkbox.disabled = False

        # Update frame slider range
        self.frame_slider.max = max(0, len(episode) - 1)
        self.frame_slider.value = 0
        self.current_frame_idx = 0

        # Display first frame
        self._display_frame(0)

    def _setup_robot(self, robot_name: str):
        """Setup robot and visualizers."""
        print(f"Loading robot: {robot_name}")

        # Load URDF
        self.urdf = load_robot_description(f"{robot_name}_description")
        self.robot = Robot.from_urdf(self.urdf)

        # Create base frame
        if hasattr(self, 'base_frame'):
            self.base_frame.remove()
        self.base_frame = self.server.scene.add_frame("/base", show_axes=False)

        # Create URDF visualizer
        if self.urdf_vis is not None:
            # Clean up old visualizer
            for handle in self.server.scene._handle_from_node_name.values():
                if "/base/" in handle.name:
                    handle.remove()

        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name="/base")

        # Create support polygon visualizer only for robots with feet
        if self.support_viz is not None:
            # Clean up old visualizer
            if hasattr(self.support_viz,
                       '_mesh_handle') and self.support_viz._mesh_handle is not None:
                self.support_viz._mesh_handle.remove()
            if hasattr(self.support_viz,
                       '_com_handle') and self.support_viz._com_handle is not None:
                self.support_viz._com_handle.remove()

        # Check if robot has foot configuration in robots_config
        try:
            from pyroki.robots_config import get_foot_link_names
            foot_links = get_foot_link_names(robot_name)
            # Only create support viz if robot has feet
            self.support_viz = SupportPolygonVisualizer(
                self.server,
                self.robot,
                root_node_name="/support_polygon",
                visible=self.show_support_polygon,
            )
            # Re-enable the checkbox
            self.show_support_checkbox.disabled = False
        except ValueError:
            # Robot doesn't have foot configuration, skip support polygon
            self.support_viz = None
            print(f"Note: Support polygon visualization not available for {robot_name}")
            # Disable the checkbox - set value first to avoid issues
            self.show_support_checkbox.value = False
            self.show_support_checkbox.disabled = True

    def _display_frame(self, frame_idx: int):
        """Display a specific frame."""
        if not self.episodes or frame_idx < 0:
            return

        episode = self.episodes[self.current_episode_idx]
        if frame_idx >= len(episode):
            return

        frame = episode[frame_idx]
        self.current_frame_idx = frame_idx

        # Update timestamp
        self.timestamp_text.value = f"{frame.timestamp:.2f}s"

        # Update robot pose
        if self.urdf_vis is not None:
            self.urdf_vis.update_cfg(frame.joint_positions)

        # Update base pose
        if self.base_frame is not None:
            self.base_frame.position = frame.root_position
            self.base_frame.wxyz = frame.root_quaternion

        # Update support polygon if available and mobile base
        if self.support_viz is not None:
            if np.any(frame.root_position != 0):
                base_pose_SE3 = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array(frame.root_quaternion)), jnp.array(frame.root_position))
                self.support_viz.update(jnp.array(frame.joint_positions), base_pose_SE3)
            else:
                # Fixed base
                self.support_viz.update(jnp.array(frame.joint_positions))
            self.com_status_text.value = self.support_viz.get_status_text()
        else:
            self.com_status_text.value = "N/A (No foot configuration)"

        # Update contact info
        num_contacts = np.sum(frame.contact_status)
        contact_names = [
            name for name, contact in zip(frame.link_names, frame.contact_status) if contact
        ]
        if contact_names:
            self.contact_text.value = f"{num_contacts} contacts: {', '.join(contact_names[:3])}..."
        else:
            self.contact_text.value = "No contacts"

    def _on_episode_change(self, _):
        """Handle episode slider change."""
        try:
            value = self.episode_slider.value
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return
            idx = int(value)
            if 0 <= idx < len(self.episodes):
                self._load_episode(idx)
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid episode slider value: {self.episode_slider.value}, error: {e}")

    def _on_frame_change(self, _):
        """Handle frame slider change.
        
        Note: We need defensive checks here because viser can sometimes send
        NaN values when the user drags the slider, especially after playback
        completes and the slider is reset.
        """
        if self.episodes and len(self.episodes) > 0:
            try:
                # Ensure we have a valid integer value
                value = self.frame_slider.value
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return
                frame_idx = int(value)
                # Clamp to valid range
                episode = self.episodes[self.current_episode_idx]
                frame_idx = max(0, min(frame_idx, len(episode) - 1))
                self._display_frame(frame_idx)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid frame slider value: {self.frame_slider.value}, error: {e}")

    def _prev_episode(self, _):
        """Go to previous episode."""
        if self.current_episode_idx > 0:
            self.episode_slider.value = self.current_episode_idx - 1

    def _next_episode(self, _):
        """Go to next episode."""
        if self.current_episode_idx < len(self.episodes) - 1:
            self.episode_slider.value = self.current_episode_idx + 1

    def _toggle_playback(self, _):
        """Toggle play/pause."""
        self.playing = not self.playing
        self.play_pause_btn.label = "Pause" if self.playing else "Play"
        self.last_update_time = time.time()

    def _toggle_support_viz(self, _):
        """Toggle support polygon visualization."""
        if self.support_viz is not None:
            self.support_viz.set_visibility(self.show_support_checkbox.value)
        # If support_viz is None, just ignore the checkbox state

    def _update_playback(self):
        """Update playback based on time."""
        if not self.playing or not self.episodes:
            return

        current_time = time.time()
        episode = self.episodes[self.current_episode_idx]

        if len(episode) <= 1:
            # Single frame episode
            return

        # Ensure current frame index is valid
        if self.current_frame_idx >= len(episode):
            self.current_frame_idx = 0
            self.last_update_time = time.time()
            return

        # Calculate time delta
        time_delta = (current_time - self.last_update_time) * self.playback_speed

        # Find next frame based on timestamp
        current_timestamp = episode[self.current_frame_idx].timestamp
        target_timestamp = current_timestamp + time_delta

        # Find the frame with closest timestamp
        next_frame_idx = self.current_frame_idx
        for i in range(self.current_frame_idx + 1, len(episode)):
            if episode[i].timestamp <= target_timestamp:
                next_frame_idx = i
            else:
                break

        # Update frame if changed
        if next_frame_idx != self.current_frame_idx:
            self.frame_slider.value = int(next_frame_idx)
            self.last_update_time = current_time

        # Loop or stop at end
        if next_frame_idx >= len(episode) - 1:
            if self.current_episode_idx < len(self.episodes) - 1:
                # Go to next episode
                self.episode_slider.value = self.current_episode_idx + 1
            else:
                # Stop at end - reset to beginning
                self.playing = False
                self.play_pause_btn.label = "Play"
                # Ensure frame is properly set before updating slider
                self.current_frame_idx = 0
                self._display_frame(0)
                self.frame_slider.value = 0
                self.last_update_time = time.time()

    def run(self):
        """Run the playback viewer."""
        print("\n=== Episode Playback Viewer ===")
        print(f"Loaded {len(self.episodes)} episodes")
        print("Controls:")
        print("  - Use Episode slider or Previous/Next buttons to navigate episodes")
        print("  - Use Frame slider to navigate within an episode")
        print("  - Click Play to automatically play through frames")
        print("  - Adjust Playback Speed to control playback rate")
        print("\nPress Ctrl+C to exit")

        try:
            while True:
                self._update_playback()
                time.sleep(0.01)    # Small sleep to prevent CPU spinning
        except KeyboardInterrupt:
            print("\nExiting playback viewer...")
