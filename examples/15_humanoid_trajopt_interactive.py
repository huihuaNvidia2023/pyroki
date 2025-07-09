"""Interactive Humanoid Trajectory Optimization

This example allows users to interactively set start and end poses for
a humanoid robot's hands while keeping feet planted, then optimizes
a trajectory between them.

Workflow:
1. Move interactive markers to desired start positions
2. Click "Capture Start Pose"
3. Move markers to desired end positions  
4. Click "Capture End Pose"
5. Click "Run Optimization" to compute trajectory
6. View the resulting trajectory playback
7. Click "Reset" to start over
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki as pk

# Import managers
from trajectory_controller import TrajectoryController


def main():
    """Main function for interactive trajectory optimization."""
    
    # Load robot
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)
    
    # Define link names
    foot_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    hand_link_names = ["left_palm_link", "right_palm_link"]
    
    # Create Viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)
    
    # Create main controller
    controller = TrajectoryController(
        robot=robot,
        urdf=urdf,
        server=server,
        foot_link_names=foot_link_names,
        hand_link_names=hand_link_names
    )
    
    print("\n=== Interactive Trajectory Optimization ===")
    print("1. Move the interactive markers to desired start positions")
    print("2. Click 'Capture Start Pose'")
    print("3. Move markers to desired end positions")
    print("4. Click 'Capture End Pose'")
    print("5. Click 'Run Optimization' to compute the trajectory")
    print("6. Use playback controls to view the result")
    print("7. Click 'Reset' to start over\n")
    
    # Main loop
    last_time = time.time()
    
    while True:
        current_time = time.time()
        
        # Update trajectory playback
        last_time = controller.update_playback(current_time, last_time)
        
        # Update visualization
        controller.update_visualization()
        
        # Small sleep to prevent CPU spinning
        time.sleep(0.01)


if __name__ == "__main__":
    main() 