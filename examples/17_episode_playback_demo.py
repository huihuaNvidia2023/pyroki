"""Demo script for episode playback viewer.

Usage:
    python 17_episode_playback_demo.py <path_to_episode_or_directory>

Examples:
    python 17_episode_playback_demo.py logs/
    python 17_episode_playback_demo.py logs/2025-01-17/panda_episode_001.parquet
"""

import sys
from pathlib import Path
import pyroki as pk


def main():
    # Get episode path from command line or use default
    if len(sys.argv) > 1:
        episode_path = sys.argv[1]
    else:
        # Try to find any episodes in logs directory
        log_dir = Path("logs")
        if log_dir.exists():
            episodes = list(log_dir.rglob("*.parquet"))
            if episodes:
                episode_path = str(log_dir)
                print(f"Found {len(episodes)} episodes in {log_dir}")
            else:
                print("No episodes found in logs/. Please provide a path to episode file(s).")
                sys.exit(1)
        else:
            print("Usage: python 17_episode_playback_demo.py <path_to_episode_or_directory>")
            sys.exit(1)

    # Create and run playback viewer
    try:
        player = pk.viewer.EpisodePlayback(
            episodes=episode_path,
            show_support_polygon=True    # Will be disabled automatically for robots without feet
        )
        player.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
