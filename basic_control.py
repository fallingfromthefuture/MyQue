#!/usr/bin/env python3
"""
Basic Control Example
=====================
Demonstrates simple autonomous robot control with the lightweight ML system.

Features:
- Autonomous operation
- Real-time monitoring
- Statistics visualization
- Save/load functionality
- Graceful shutdown

Usage:
    python3 basic_control.py [--load PATH] [--save PATH] [--frequency HZ]

Author: RoboOS AI Team
License: MIT
"""

import sys
import time
import argparse
import signal
import numpy as np
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

# Import the main controller
sys.path.insert(0, str(Path(__file__).parent.parent))
from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action
)


class MonitoredController(LightweightRobotController):
    """
    Enhanced controller with monitoring and visualization
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'total_actions': 0,
            'total_observations': 0,
            'objects_detected': 0,
            'average_fps': 0.0,
            'last_fps_update': time.time(),
            'fps_samples': []
        }
    
    def get_observation(self):
        """Get observation and update stats"""
        obs = super().get_observation()
        
        if obs is not None:
            self.stats['total_observations'] += 1
            
            # Update FPS
            current_time = time.time()
            if current_time - self.stats['last_fps_update'] > 0.1:
                self.stats['fps_samples'].append(
                    1.0 / (current_time - self.stats['last_fps_update'])
                )
                if len(self.stats['fps_samples']) > 30:
                    self.stats['fps_samples'].pop(0)
                self.stats['average_fps'] = np.mean(self.stats['fps_samples'])
                self.stats['last_fps_update'] = current_time
        
        return obs
    
    def execute_action(self, action):
        """Execute action and update stats"""
        super().execute_action(action)
        self.stats['total_actions'] += 1
    
    def compute_reward(self, observation, action):
        """Enhanced reward with object detection"""
        # Detect objects
        objects = self.object_detector.detect(observation.image)
        
        if len(objects) > 0:
            self.stats['objects_detected'] += len(objects)
        
        # Base reward
        reward = 0.0
        
        # Reward for finding objects
        if len(objects) > 0:
            reward += 1.0
            
            # Reward for centering largest object
            largest = max(objects, key=lambda o: o['area'])
            center_x = largest['center'][0]
            image_center = observation.image.shape[1] / 2
            
            # Distance from center (0 = perfect, 1 = edge)
            distance = abs(center_x - image_center) / (observation.image.shape[1] / 2)
            centering_reward = 1.0 - distance
            reward += centering_reward
        
        # Penalty for large velocities (encourage smooth motion)
        velocity_penalty = 0.1 * np.linalg.norm(action.linear_velocity)
        reward -= velocity_penalty
        
        return reward
    
    def get_stats(self):
        """Get current statistics"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'uptime': uptime,
            'total_actions': self.stats['total_actions'],
            'total_observations': self.stats['total_observations'],
            'objects_detected': self.stats['objects_detected'],
            'experiences': len(self.replay_buffer),
            'learning_updates': self.learner.update_counter,
            'fps': self.stats['average_fps']
        }


def draw_stats(image, stats):
    """Draw statistics overlay on image"""
    if not HAS_CV2:
        return image
    
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Semi-transparent background
    cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    
    # Title
    cv2.putText(result, "Robot Control - Autonomous Mode", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Stats
    y = 65
    stats_text = [
        f"Uptime: {stats['uptime']:.1f}s",
        f"FPS: {stats['fps']:.1f}",
        f"Observations: {stats['total_observations']}",
        f"Actions: {stats['total_actions']}",
        f"Objects Detected: {stats['objects_detected']}",
        f"Experiences: {stats['experiences']}",
        f"Learning Updates: {stats['learning_updates']}"
    ]
    
    for text in stats_text:
        cv2.putText(result, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
    
    # Status
    if stats['experiences'] >= 32:
        status = "LEARNING"
        color = (0, 255, 0)
    else:
        status = "COLLECTING"
        color = (255, 255, 0)
    
    cv2.putText(result, f"Status: {status}", (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result


def print_stats(stats):
    """Print statistics to console"""
    print("\033[2J\033[H")  # Clear screen
    print("=" * 70)
    print("ROBOT CONTROL - AUTONOMOUS MODE")
    print("=" * 70)
    print()
    print(f"Uptime:           {stats['uptime']:.1f} seconds")
    print(f"FPS:              {stats['fps']:.1f}")
    print(f"Observations:     {stats['total_observations']}")
    print(f"Actions:          {stats['total_actions']}")
    print(f"Objects Detected: {stats['objects_detected']}")
    print()
    print("LEARNING")
    print("-" * 70)
    print(f"Experiences:      {stats['experiences']}")
    print(f"Learning Updates: {stats['learning_updates']}")
    
    if stats['experiences'] >= 32:
        print(f"Status:           {'LEARNING' if stats['learning_updates'] > 0 else 'INITIALIZING'}")
    else:
        print(f"Status:           COLLECTING ({stats['experiences']}/32 needed)")
    
    print()
    print("Press Ctrl+C to stop and save")


def main():
    """Main function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Basic Robot Control Example')
    parser.add_argument('--load', type=str, default='saved_controller',
                       help='Path to load controller state from')
    parser.add_argument('--save', type=str, default='saved_controller',
                       help='Path to save controller state to')
    parser.add_argument('--frequency', type=int, default=30,
                       help='Control loop frequency in Hz')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable graphical display')
    parser.add_argument('--stats-interval', type=float, default=1.0,
                       help='Statistics update interval in seconds')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("LIGHTWEIGHT ML ROBOT CONTROL - BASIC AUTONOMOUS MODE")
    print("=" * 70)
    print()
    print("This example demonstrates basic autonomous robot control with")
    print("online learning from experience.")
    print()
    
    # Configuration
    config = {
        'control_frequency': args.frequency,
        'learning_rate': 0.001
    }
    
    print("Configuration:")
    print(f"  Control frequency: {args.frequency} Hz")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Display: {'Enabled' if not args.no_display and HAS_CV2 else 'Disabled'}")
    print()
    
    # Create controller
    print("Initializing controller...")
    controller = MonitoredController(config)
    
    # Try to load previous training
    load_path = Path(args.load)
    if load_path.exists():
        try:
            controller.load(str(load_path))
            print(f"✓ Loaded previous training from: {load_path}")
            print(f"  - Experiences: {len(controller.replay_buffer)}")
            print(f"  - Learning updates: {controller.learner.update_counter}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            print("  Starting fresh")
    else:
        print("No previous training found, starting fresh")
    
    print()
    print("Starting robot...")
    
    # Setup signal handler for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start controller
    controller.start()
    
    print("✓ Robot started!")
    print()
    
    try:
        last_stats_update = time.time()
        
        while not shutdown_requested:
            current_time = time.time()
            
            # Update statistics
            if current_time - last_stats_update >= args.stats_interval:
                stats = controller.get_stats()
                
                if not args.no_display and HAS_CV2:
                    # Display with visualization
                    obs = controller.current_observation
                    if obs is not None:
                        display_image = draw_stats(obs.image, stats)
                        cv2.imshow('Robot Control - Autonomous Mode', display_image)
                        
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC
                            shutdown_requested = True
                else:
                    # Console-only display
                    print_stats(stats)
                
                last_stats_update = current_time
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n\n" + "=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        
        # Stop controller
        print("\nStopping robot...")
        controller.stop()
        print("✓ Robot stopped")
        
        # Close display
        if HAS_CV2:
            cv2.destroyAllWindows()
        
        # Save state
        save_path = Path(args.save)
        print(f"\nSaving controller state to: {save_path}")
        try:
            controller.save(str(save_path))
            print("✓ State saved successfully")
        except Exception as e:
            print(f"✗ Failed to save: {e}")
        
        # Print final summary
        final_stats = controller.get_stats()
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total runtime:          {final_stats['uptime']:.1f} seconds")
        print(f"Total observations:     {final_stats['total_observations']}")
        print(f"Total actions:          {final_stats['total_actions']}")
        print(f"Objects detected:       {final_stats['objects_detected']}")
        print(f"Experiences collected:  {final_stats['experiences']}")
        print(f"Learning updates:       {final_stats['learning_updates']}")
        print(f"Average FPS:            {final_stats['fps']:.1f}")
        print()
        print("Next time you run this, the robot will continue learning from")
        print("where it left off!")
        print()


if __name__ == '__main__':
    main()
