"""GUI control panel for interactive trajectory optimization."""

from typing import Callable
import viser
from trajectory_state import TrajectoryStates


class ControlPanel:
    """GUI controls for the interactive workflow."""
    
    def __init__(self, server: viser.ViserServer):
        """Initialize control panel."""
        self.server = server
        
        # Create GUI elements
        with server.gui.add_folder("Trajectory Setup"):
            self.state_text = server.gui.add_text("State", "Setup Start Pose", disabled=True)
            self.capture_start_btn = server.gui.add_button("Capture Start Pose")
            self.capture_end_btn = server.gui.add_button("Capture End Pose")
            self.run_btn = server.gui.add_button("Run Optimization")
            self.reset_btn = server.gui.add_button("Reset")
            
        with server.gui.add_folder("Trajectory Parameters"):
            self.timesteps_slider = server.gui.add_slider(
                "Timesteps", min=10, max=100, step=1, initial_value=30
            )
            self.dt_slider = server.gui.add_slider(
                "Time Step (s)", min=0.01, max=0.1, step=0.01, initial_value=0.05
            )
            self.duration_text = server.gui.add_text("Duration", "1.5s", disabled=True)
            
        with server.gui.add_folder("COM Support"):
            self.com_weight_slider = server.gui.add_slider(
                "Weight", min=0.0, max=1000.0, step=10.0, initial_value=500.0
            )
            self.com_margin_slider = server.gui.add_slider(
                "Margin (m)", min=0.0, max=0.1, step=0.01, initial_value=0.0
            )
            
        with server.gui.add_folder("Base Constraints"):
            self.fix_x = server.gui.add_checkbox("Fix X", False)
            self.fix_y = server.gui.add_checkbox("Fix Y", False) 
            self.fix_z = server.gui.add_checkbox("Fix Z", False)  # Changed to False
            self.fix_roll = server.gui.add_checkbox("Fix Roll", False)  # Changed to False
            self.fix_pitch = server.gui.add_checkbox("Fix Pitch", False)  # Changed to False
            self.fix_yaw = server.gui.add_checkbox("Fix Yaw", False)
            
        with server.gui.add_folder("Playback Controls"):
            self.timestep_slider = server.gui.add_slider(
                "Timestep", min=0, max=29, step=1, initial_value=0
            )
            self.playing = server.gui.add_checkbox("Playing", initial_value=False)
            self.speed_slider = server.gui.add_slider(
                "Playback Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
            )
            
        with server.gui.add_folder("Visualization"):
            self.show_support = server.gui.add_checkbox("Show Support Polygon", True)
            self.show_markers = server.gui.add_checkbox("Show Pose Markers", True)
            self.com_status_text = server.gui.add_text("COM Status", "N/A", disabled=True)
            
        # Update duration when timesteps or dt changes
        self.timesteps_slider.on_update(self._update_duration)
        self.dt_slider.on_update(self._update_duration)
        
        # Initially disable certain buttons
        self.update_ui_state(TrajectoryStates.SETUP_START)
        
    def _update_duration(self, _=None):
        """Update duration display based on timesteps and dt."""
        duration = self.timesteps_slider.value * self.dt_slider.value
        self.duration_text.value = f"{duration:.1f}s"
        
    def update_ui_state(self, state: TrajectoryStates) -> None:
        """Update button availability based on current state."""
        # Update state text
        state_texts = {
            TrajectoryStates.SETUP_START: "Setup Start Pose",
            TrajectoryStates.SETUP_END: "Setup End Pose", 
            TrajectoryStates.READY: "Ready to Optimize",
            TrajectoryStates.OPTIMIZING: "Optimizing...",
            TrajectoryStates.PLAYBACK: "Playing Trajectory"
        }
        self.state_text.value = state_texts[state]
        
        # Update button states
        self.capture_start_btn.disabled = state != TrajectoryStates.SETUP_START
        self.capture_end_btn.disabled = state != TrajectoryStates.SETUP_END
        self.run_btn.disabled = state != TrajectoryStates.READY
        self.reset_btn.disabled = state == TrajectoryStates.OPTIMIZING
        
        # Update playback controls
        playback_enabled = state == TrajectoryStates.PLAYBACK
        self.timestep_slider.disabled = not playback_enabled
        self.playing.disabled = not playback_enabled
        self.speed_slider.disabled = not playback_enabled
        
    def get_base_constraints(self) -> tuple:
        """Get current base constraint settings."""
        fix_position = (self.fix_x.value, self.fix_y.value, self.fix_z.value)
        fix_orientation = (self.fix_roll.value, self.fix_pitch.value, self.fix_yaw.value)
        return fix_position, fix_orientation
        
    def update_timestep_range(self, max_timestep: int) -> None:
        """Update timestep slider range after optimization."""
        self.timestep_slider.max = max_timestep - 1
        self.timestep_slider.value = 0 