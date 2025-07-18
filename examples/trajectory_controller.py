"""Main controller for interactive trajectory optimization."""

from typing import Dict, List
import numpy as np
import pyroki as pk
import viser
import pyroki_snippets as pks

from trajectory_state import TrajectoryState, TrajectoryStates
from interactive_marker_manager import InteractiveMarkerManager
from visualization_manager import VisualizationManager
from control_panel import ControlPanel


class TrajectoryController:
    """Controls the interactive trajectory optimization workflow."""
    
    def __init__(self, robot: pk.Robot, urdf, server: viser.ViserServer,
                 foot_link_names: List[str], hand_link_names: List[str]):
        """
        Initialize trajectory controller.
        
        Args:
            robot: PyRoKi robot instance
            urdf: URDF model
            server: Viser server instance
            foot_link_names: List of foot link names
            hand_link_names: List of hand link names
        """
        self.robot = robot
        self.urdf = urdf
        self.server = server
        self.foot_link_names = foot_link_names
        self.hand_link_names = hand_link_names
        self.all_link_names = foot_link_names + hand_link_names
        
        # Initialize state
        self.state = TrajectoryState()
        
        # Initialize managers
        self.marker_manager = InteractiveMarkerManager(server, self.all_link_names)
        self.viz_manager = VisualizationManager(server, robot, urdf)
        self.control_panel = ControlPanel(server)
        
        # Connect button callbacks
        self.control_panel.capture_start_btn.on_click(self.capture_start_pose)
        self.control_panel.capture_end_btn.on_click(self.capture_end_pose)
        self.control_panel.run_btn.on_click(self.run_optimization)
        self.control_panel.reset_btn.on_click(self.reset)
        
        # Connect visualization callbacks
        self.control_panel.show_support.on_update(
            lambda _: self.viz_manager.set_support_polygon_visible(
                self.control_panel.show_support.value
            )
        )
        self.control_panel.show_labels.on_update(
            lambda _: self.viz_manager.set_labels_visible(
                self.control_panel.show_labels.value
            )
        )
        self.control_panel.show_markers.on_update(
            lambda _: self._update_marker_visibility()
        )
        
        # Initialize robot pose
        self.initial_base_pos = np.array([0.0, 0.0, 0.75])
        self.initial_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Get initial marker positions for all links
        initial_positions, initial_wxyzs = self.marker_manager.get_poses()
        
        # Solve initial IK to get robot into standing pose
        print("Setting up initial standing pose...")
        all_positions = np.array([initial_positions[name] for name in self.all_link_names])
        all_wxyzs = np.array([initial_wxyzs[name] for name in self.all_link_names])
        
        # Use solve_ik_with_multiple_targets_and_base for initial pose
        base_pos, base_wxyz, joint_cfg = pks.solve_ik_with_multiple_targets_and_base(
            robot=self.robot,
            target_link_names=self.all_link_names,  # Use all links, not just feet
            target_positions=all_positions,
            target_wxyzs=all_wxyzs,
            fix_base_position=(False, False, False),
            fix_base_orientation=(False, False, False),
            prev_pos=self.initial_base_pos,
            prev_wxyz=self.initial_base_wxyz,
            prev_cfg=np.array(robot.joint_var_cls(0).default_factory()),
        )
        
        # Update initial values
        self.initial_base_pos = np.array(base_pos)
        self.initial_base_wxyz = np.array(base_wxyz)
        self.initial_cfg = np.array(joint_cfg)
        
        # Set initial robot pose
        self.viz_manager.update_robot_pose(
            self.initial_base_pos,
            self.initial_base_wxyz,
            self.initial_cfg
        )
        
    def capture_start_pose(self, _=None) -> None:
        """Capture current marker positions as start pose."""
        positions, wxyzs = self.marker_manager.get_poses()
        self.state.capture_start_poses(positions, wxyzs)
        
        # Update UI
        self.control_panel.update_ui_state(self.state.current_state)
        
        # Show captured poses if checkbox is enabled
        if self.control_panel.show_markers.value:
            self.viz_manager.show_pose_markers(positions, wxyzs)
        
        print("Start pose captured!")
        
    def capture_end_pose(self, _=None) -> None:
        """Capture current marker positions as end pose."""
        positions, wxyzs = self.marker_manager.get_poses()
        self.state.capture_end_poses(positions, wxyzs)
        
        # Update UI
        self.control_panel.update_ui_state(self.state.current_state)
        
        # Show both start and end poses if checkbox is enabled
        if self.control_panel.show_markers.value:
            self.viz_manager.show_pose_markers(
                self.state.start_positions,
                self.state.start_wxyzs,
                positions,
                wxyzs
            )
        
        print("End pose captured!")
        
    def run_optimization(self, _=None) -> None:
        """Run trajectory optimization with captured poses."""
        if self.state.current_state != TrajectoryStates.READY:
            return
            
        # Update state
        self.state.current_state = TrajectoryStates.OPTIMIZING
        self.control_panel.update_ui_state(self.state.current_state)
        
        # Get parameters from UI
        timesteps = self.control_panel.timesteps_slider.value
        dt = self.control_panel.dt_slider.value
        com_weight = self.control_panel.com_weight_slider.value
        com_margin = self.control_panel.com_margin_slider.value
        fix_position, fix_orientation = self.control_panel.get_base_constraints()
        
        # Prepare arrays for optimization
        foot_positions = np.array([self.state.start_positions[name] for name in self.foot_link_names])
        foot_wxyzs = np.array([self.state.start_wxyzs[name] for name in self.foot_link_names])
        hand_start_positions = np.array([self.state.start_positions[name] for name in self.hand_link_names])
        hand_start_wxyzs = np.array([self.state.start_wxyzs[name] for name in self.hand_link_names])
        hand_end_positions = np.array([self.state.end_positions[name] for name in self.hand_link_names])
        hand_end_wxyzs = np.array([self.state.end_wxyzs[name] for name in self.hand_link_names])
        
        try:
            print("Running trajectory optimization...")
            
            # Run optimization
            base_positions, base_wxyzs, joint_cfgs = pks.solve_trajopt_with_base(
                robot=self.robot,
                foot_link_names=self.foot_link_names,
                hand_link_names=self.hand_link_names,
                foot_positions=foot_positions,
                foot_wxyzs=foot_wxyzs,
                hand_start_positions=hand_start_positions,
                hand_start_wxyzs=hand_start_wxyzs,
                hand_end_positions=hand_end_positions,
                hand_end_wxyzs=hand_end_wxyzs,
                fix_base_position=fix_position,
                fix_base_orientation=fix_orientation,
                timesteps=timesteps,
                dt=dt,
                prev_pos=self.initial_base_pos,
                prev_wxyz=self.initial_base_wxyz,
                com_support_weight=com_weight,
                com_support_margin=com_margin,
            )
            
            print("Trajectory optimization complete!")
            
            # Ensure numpy arrays
            base_positions = np.array(base_positions)
            base_wxyzs = np.array(base_wxyzs)
            joint_cfgs = np.array(joint_cfgs)
            
            # Store results
            self.state.timesteps = timesteps
            self.state.dt = dt
            self.state.set_trajectory_data(base_positions, base_wxyzs, joint_cfgs)
            
            # Update UI
            self.control_panel.update_ui_state(self.state.current_state)
            self.control_panel.update_timestep_range(timesteps)
            
            # Start playback
            self.control_panel.playing.value = True
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            self.state.current_state = TrajectoryStates.READY
            self.control_panel.update_ui_state(self.state.current_state)
            
    def reset(self, _=None) -> None:
        """Reset to initial state."""
        print("Resetting...")
        
        # Reset state
        self.state.reset()
        
        # Reset UI
        self.control_panel.update_ui_state(self.state.current_state)
        self.control_panel.timestep_slider.value = 0
        self.control_panel.playing.value = False
        
        # Clear visualizations
        self.viz_manager._clear_pose_markers()
        
        # Reset support polygon to avoid handle issues
        self.viz_manager.reset_support_polygon()
        
        # Sync support polygon visibility with checkbox
        self.viz_manager.set_support_polygon_visible(self.control_panel.show_support.value)
        
        # Clear old markers
        for marker in self.marker_manager.markers.values():
            marker.remove()
            
        # Reset marker positions to defaults
        self.marker_manager = InteractiveMarkerManager(self.server, self.all_link_names)
        
        # No need to solve IK again - just reuse the initial values
        # since markers are at the same default positions
        print("Resetting to initial standing pose...")
        
        # Set robot to initial pose
        self.viz_manager.update_robot_pose(
            self.initial_base_pos,
            self.initial_base_wxyz,
            self.initial_cfg
        )
        
    def update_playback(self, current_time: float, last_time: float) -> float:
        """Update trajectory playback based on time."""
        if (self.state.current_state != TrajectoryStates.PLAYBACK or 
            not self.control_panel.playing.value):
            return last_time
            
        # Calculate timestep update
        time_delta = (current_time - last_time) * self.control_panel.speed_slider.value
        timestep_delta = int(time_delta / self.state.dt)
        
        if timestep_delta > 0:
            # Update timestep
            current_step = self.control_panel.timestep_slider.value
            new_step = (current_step + timestep_delta) % self.state.timesteps
            self.control_panel.timestep_slider.value = new_step
            
            return current_time
            
        return last_time
        
    def _update_marker_visibility(self) -> None:
        """Update marker visibility based on checkbox state."""
        show_markers = self.control_panel.show_markers.value
        
        if not show_markers:
            # Hide all markers
            self.viz_manager._clear_pose_markers()
        else:
            # Show markers if we have pose data
            if self.state.start_positions:
                self.viz_manager.show_pose_markers(
                    self.state.start_positions,
                    self.state.start_wxyzs,
                    self.state.end_positions if hasattr(self.state, 'end_positions') and self.state.end_positions else None,
                    self.state.end_wxyzs if hasattr(self.state, 'end_wxyzs') and self.state.end_wxyzs else None
                )
    
    def update_visualization(self) -> None:
        """Update robot visualization based on current timestep."""
        if self.state.current_state != TrajectoryStates.PLAYBACK:
            return
            
        t = self.control_panel.timestep_slider.value
        
        # Update robot pose
        self.viz_manager.update_robot_pose(
            self.state.base_positions[t],
            self.state.base_wxyzs[t],
            self.state.joint_cfgs[t]
        )
        
        # Update COM status
        self.control_panel.com_status_text.value = self.viz_manager.get_com_status() 