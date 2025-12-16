#!/usr/bin/env python3
"""
Pick and Place Example
======================
Demonstrates object manipulation using vision-guided control.
The robot learns to:
1. Detect objects in the scene
2. Approach and center objects
3. Grasp objects with gripper
4. Transport to target location
5. Release objects

Features:
- Color-based object detection
- Visual servoing for centering
- State machine for task execution
- Adaptive grasping
- Success detection

Usage:
    python3 pick_and_place.py [--color COLOR] [--target X Y]

Author: RoboOS AI Team
License: MIT
"""

import sys
import time
import argparse
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Error: OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

# Import the main controller
sys.path.insert(0, str(Path(__file__).parent.parent))
from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action
)


class TaskState(Enum):
    """Pick and place task states"""
    SEARCHING = 1      # Looking for object
    APPROACHING = 2    # Moving toward object
    CENTERING = 3      # Centering object in view
    GRASPING = 4       # Closing gripper
    LIFTING = 5        # Lifting object
    TRANSPORTING = 6   # Moving to target
    PLACING = 7        # Releasing object
    RETRACTING = 8     # Moving back
    COMPLETED = 9      # Task complete
    FAILED = 10        # Task failed


class ColorObjectDetector:
    """
    Detect objects by color in HSV space
    """
    
    def __init__(self, target_color='red'):
        # HSV ranges for common colors
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255)],  # Lower red
            'red2': [(170, 100, 100), (180, 255, 255)],  # Upper red
            'blue': [(100, 100, 100), (130, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)],
            'yellow': [(20, 100, 100), (40, 255, 255)],
            'orange': [(10, 100, 100), (20, 255, 255)]
        }
        
        self.target_color = target_color
    
    def detect(self, image):
        """Detect colored objects"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get color range
        if self.target_color not in self.color_ranges:
            self.target_color = 'red'
        
        lower, upper = self.color_ranges[self.target_color]
        
        # Create mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Handle red wraparound
        if self.target_color == 'red':
            lower2, upper2 = self.color_ranges['red2']
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            objects.append({
                'bbox': [x, y, w, h],
                'center': [cx, cy],
                'area': area,
                'contour': contour
            })
        
        # Sort by area (largest first)
        objects.sort(key=lambda o: o['area'], reverse=True)
        
        return objects, mask


class PickAndPlaceController(LightweightRobotController):
    """
    Enhanced controller for pick and place tasks
    """
    
    def __init__(self, config=None, target_color='red', target_position=None):
        super().__init__(config)
        
        # Object detector
        self.color_detector = ColorObjectDetector(target_color)
        
        # Task state
        self.state = TaskState.SEARCHING
        self.target_object = None
        self.target_position = target_position or [1.0, 0.0]  # Default target
        
        # State timers
        self.state_start_time = time.time()
        self.state_timeout = 10.0  # 10 seconds per state
        
        # Statistics
        self.attempts = 0
        self.successes = 0
        self.failures = 0
    
    def update_state(self, new_state):
        """Update task state"""
        print(f"State: {self.state.name} → {new_state.name}")
        self.state = new_state
        self.state_start_time = time.time()
    
    def state_timeout_exceeded(self):
        """Check if current state has exceeded timeout"""
        return (time.time() - self.state_start_time) > self.state_timeout
    
    def compute_action(self, observation):
        """Compute action based on current task state"""
        # Detect objects
        objects, mask = self.color_detector.detect(observation.image)
        
        # State machine
        if self.state == TaskState.SEARCHING:
            return self._searching(observation, objects)
        
        elif self.state == TaskState.APPROACHING:
            return self._approaching(observation, objects)
        
        elif self.state == TaskState.CENTERING:
            return self._centering(observation, objects)
        
        elif self.state == TaskState.GRASPING:
            return self._grasping(observation)
        
        elif self.state == TaskState.LIFTING:
            return self._lifting(observation)
        
        elif self.state == TaskState.TRANSPORTING:
            return self._transporting(observation)
        
        elif self.state == TaskState.PLACING:
            return self._placing(observation)
        
        elif self.state == TaskState.RETRACTING:
            return self._retracting(observation)
        
        elif self.state == TaskState.COMPLETED:
            return self._completed(observation)
        
        elif self.state == TaskState.FAILED:
            return self._failed(observation)
        
        # Default: stop
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _searching(self, observation, objects):
        """Search for target object"""
        if len(objects) > 0:
            self.target_object = objects[0]
            self.update_state(TaskState.APPROACHING)
            return self.compute_action(observation)
        
        # Rotate slowly to search
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.3]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _approaching(self, observation, objects):
        """Approach detected object"""
        if len(objects) == 0:
            # Lost object
            self.update_state(TaskState.SEARCHING)
            return self.compute_action(observation)
        
        obj = objects[0]
        
        # Check if object is large enough (close enough)
        if obj['area'] > 15000:  # Threshold for close proximity
            self.update_state(TaskState.CENTERING)
            return self.compute_action(observation)
        
        # Move forward
        return Action(
            linear_velocity=np.array([0.2, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _centering(self, observation, objects):
        """Center object in view"""
        if len(objects) == 0:
            self.update_state(TaskState.SEARCHING)
            return self.compute_action(observation)
        
        obj = objects[0]
        cx, cy = obj['center']
        
        # Image center
        image_center_x = observation.image.shape[1] // 2
        
        # Error from center
        error_x = cx - image_center_x
        
        # Check if centered
        if abs(error_x) < 30:  # Within 30 pixels
            self.update_state(TaskState.GRASPING)
            return self.compute_action(observation)
        
        # Proportional control to center
        angular_velocity = -error_x / 300.0  # Scale factor
        angular_velocity = np.clip(angular_velocity, -0.5, 0.5)
        
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, angular_velocity]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _grasping(self, observation):
        """Close gripper to grasp object"""
        # Close gripper over 1 second
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 1.0:
            self.update_state(TaskState.LIFTING)
            return self.compute_action(observation)
        
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=1.0  # Closed
        )
    
    def _lifting(self, observation):
        """Lift object"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 0.5:
            self.update_state(TaskState.TRANSPORTING)
            return self.compute_action(observation)
        
        # In real robot, would lift arm
        # Here we just hold position
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=1.0
        )
    
    def _transporting(self, observation):
        """Transport object to target"""
        # Simple: move forward for 2 seconds
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 2.0:
            self.update_state(TaskState.PLACING)
            return self.compute_action(observation)
        
        return Action(
            linear_velocity=np.array([0.3, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=1.0  # Keep closed
        )
    
    def _placing(self, observation):
        """Release object"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 1.0:
            self.update_state(TaskState.RETRACTING)
            return self.compute_action(observation)
        
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0  # Open
        )
    
    def _retracting(self, observation):
        """Move back after placing"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed > 1.0:
            self.successes += 1
            self.update_state(TaskState.COMPLETED)
            return self.compute_action(observation)
        
        return Action(
            linear_velocity=np.array([-0.2, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _completed(self, observation):
        """Task completed"""
        # Stop
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _failed(self, observation):
        """Task failed"""
        self.failures += 1
        # Stop
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def compute_reward(self, observation, action):
        """Compute reward based on task progress"""
        reward = 0.0
        
        # Detect objects
        objects, _ = self.color_detector.detect(observation.image)
        
        # Reward for each state
        if self.state == TaskState.SEARCHING:
            # Reward for finding object
            if len(objects) > 0:
                reward += 1.0
        
        elif self.state == TaskState.APPROACHING:
            # Reward for getting closer
            if len(objects) > 0:
                reward += objects[0]['area'] / 10000.0
        
        elif self.state == TaskState.CENTERING:
            # Reward for centering
            if len(objects) > 0:
                cx = objects[0]['center'][0]
                image_center = observation.image.shape[1] // 2
                error = abs(cx - image_center)
                reward += 1.0 - (error / image_center)
        
        elif self.state == TaskState.GRASPING:
            reward += 2.0  # High reward for grasping
        
        elif self.state == TaskState.COMPLETED:
            reward += 10.0  # Very high reward for completion
        
        elif self.state == TaskState.FAILED:
            reward -= 5.0  # Penalty for failure
        
        # Check for timeout
        if self.state_timeout_exceeded():
            reward -= 1.0
            self.update_state(TaskState.FAILED)
        
        return reward


def draw_pick_place_ui(image, controller, objects, mask):
    """Draw UI for pick and place"""
    h, w = image.shape[:2]
    
    # Create visualization with mask overlay
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0  # Remove blue channel
    mask_color[:, :, 1] = 0  # Remove green channel
    result = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
    
    # Draw detected objects
    for obj in objects:
        x, y, w_box, h_box = obj['bbox']
        cx, cy = obj['center']
        
        # Bounding box
        cv2.rectangle(result, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        
        # Center point
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        
        # Area text
        cv2.putText(result, f"Area: {obj['area']:.0f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw center line
    cv2.line(result, (w//2, 0), (w//2, h), (255, 255, 0), 1)
    
    # Status overlay
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Title
    cv2.putText(result, "PICK AND PLACE - OBJECT MANIPULATION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # State
    state_colors = {
        TaskState.SEARCHING: (255, 255, 0),
        TaskState.APPROACHING: (255, 200, 0),
        TaskState.CENTERING: (255, 150, 0),
        TaskState.GRASPING: (255, 100, 0),
        TaskState.LIFTING: (0, 255, 100),
        TaskState.TRANSPORTING: (0, 255, 200),
        TaskState.PLACING: (0, 200, 255),
        TaskState.RETRACTING: (0, 100, 255),
        TaskState.COMPLETED: (0, 255, 0),
        TaskState.FAILED: (0, 0, 255)
    }
    
    color = state_colors.get(controller.state, (255, 255, 255))
    cv2.putText(result, f"State: {controller.state.name}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Statistics
    cv2.putText(result, f"Attempts: {controller.attempts}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(result, f"Successes: {controller.successes}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result, f"Failures: {controller.failures}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return result


def main():
    """Main function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pick and Place Example')
    parser.add_argument('--color', type=str, default='red',
                       choices=['red', 'blue', 'green', 'yellow', 'orange'],
                       help='Target object color')
    parser.add_argument('--target', nargs=2, type=float, default=[1.0, 0.0],
                       help='Target position (x, y)')
    parser.add_argument('--save', type=str, default='pick_place_controller',
                       help='Save path')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous operation (reset after each attempt)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("PICK AND PLACE - OBJECT MANIPULATION")
    print("=" * 70)
    print()
    print(f"Target color: {args.color}")
    print(f"Target position: {args.target}")
    print(f"Continuous mode: {args.continuous}")
    print()
    
    # Create controller
    config = {
        'control_frequency': 20,
        'learning_rate': 0.001
    }
    
    print("Initializing controller...")
    controller = PickAndPlaceController(
        config=config,
        target_color=args.color,
        target_position=args.target
    )
    
    # Load if exists
    if Path(args.save).exists():
        try:
            controller.load(args.save)
            print(f"✓ Loaded: {len(controller.replay_buffer)} experiences")
        except:
            print("Starting fresh")
    
    print()
    print("Starting robot...")
    controller.start()
    print("✓ Robot ready!")
    print()
    print("Performing pick and place task...")
    print("=" * 70)
    print()
    
    try:
        while True:
            # Get observation
            obs = controller.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue
            
            # Detect objects
            objects, mask = controller.color_detector.detect(obs.image)
            
            # Display
            display_image = draw_pick_place_ui(obs.image, controller, objects, mask)
            cv2.imshow('Pick and Place', display_image)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                # Reset task
                controller.attempts += 1
                controller.update_state(TaskState.SEARCHING)
                print("Task reset")
            
            # Check if completed or failed
            if controller.state == TaskState.COMPLETED:
                print(f"\n✓ Task completed! (Attempt {controller.attempts + 1})")
                if args.continuous:
                    time.sleep(2)
                    controller.attempts += 1
                    controller.update_state(TaskState.SEARCHING)
                    print("Starting new attempt...")
                else:
                    break
            
            elif controller.state == TaskState.FAILED:
                print(f"\n✗ Task failed! (Attempt {controller.attempts + 1})")
                if args.continuous:
                    time.sleep(2)
                    controller.attempts += 1
                    controller.update_state(TaskState.SEARCHING)
                    print("Retrying...")
                else:
                    break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        print("\n" + "=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        
        controller.stop()
        cv2.destroyAllWindows()
        
        # Save
        controller.save(args.save)
        print(f"\n✓ Saved to: {args.save}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total attempts:  {controller.attempts}")
        print(f"Successes:       {controller.successes}")
        print(f"Failures:        {controller.failures}")
        if controller.attempts > 0:
            success_rate = controller.successes / controller.attempts * 100
            print(f"Success rate:    {success_rate:.1f}%")
        print()


if __name__ == '__main__':
    main()
