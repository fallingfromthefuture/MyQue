#!/usr/bin/env python3
"""
Autonomous Navigation Example
==============================
Demonstrates autonomous navigation with obstacle avoidance using vision
and/or ultrasonic sensors.

Features:
- Goal-directed navigation
- Dynamic obstacle avoidance
- Path planning (potential fields)
- Recovery behaviors
- Visual SLAM (basic)

Usage:
    python3 navigation.py [--goal X Y] [--avoid-obstacles]

Author: RoboOS AI Team
License: MIT
"""

import sys
import time
import argparse
from pathlib import Path
from enum import Enum
import numpy as np
from collections import deque

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Error: OpenCV required")
    sys.exit(1)

# Import the main controller
sys.path.insert(0, str(Path(__file__).parent.parent))
from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action
)


class NavigationState(Enum):
    """Navigation states"""
    MOVING_TO_GOAL = 1
    AVOIDING_OBSTACLE = 2
    RECOVERING = 3
    REACHED_GOAL = 4
    STUCK = 5


class ObstacleDetector:
    """
    Detect obstacles using vision
    """
    
    def __init__(self, detection_distance=100):
        self.detection_distance = detection_distance
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    def detect(self, image):
        """Detect obstacles in image"""
        h, w = image.shape[:2]
        
        # Use bottom half of image (ground plane)
        bottom_half = image[h//2:, :]
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(bottom_half)
        
        # Threshold
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Divide into sectors (left, center, right)
        sector_width = w // 3
        
        sectors = {
            'left': fg_mask[:, :sector_width],
            'center': fg_mask[:, sector_width:2*sector_width],
            'right': fg_mask[:, 2*sector_width:]
        }
        
        # Calculate obstacle density in each sector
        obstacle_density = {}
        for name, sector in sectors.items():
            density = np.sum(sector) / (sector.size * 255)
            obstacle_density[name] = density
        
        # Determine clear direction
        min_density = min(obstacle_density.values())
        clear_sectors = [k for k, v in obstacle_density.items() if v == min_density]
        
        return obstacle_density, clear_sectors[0] if clear_sectors else None, fg_mask


class PotentialFieldNavigator:
    """
    Navigate using artificial potential fields
    """
    
    def __init__(self, goal_position, attractive_gain=1.0, repulsive_gain=0.5):
        self.goal = np.array(goal_position)
        self.ka = attractive_gain
        self.kr = repulsive_gain
        self.influence_distance = 1.0  # Obstacles influence within 1m
    
    def compute_force(self, current_position, obstacles):
        """Compute total force on robot"""
        pos = np.array(current_position)
        
        # Attractive force toward goal
        to_goal = self.goal - pos
        distance_to_goal = np.linalg.norm(to_goal)
        
        if distance_to_goal > 0:
            attractive_force = self.ka * (to_goal / distance_to_goal)
        else:
            attractive_force = np.zeros(2)
        
        # Repulsive force from obstacles
        repulsive_force = np.zeros(2)
        
        for obs_pos in obstacles:
            to_obstacle = pos - np.array(obs_pos)
            distance = np.linalg.norm(to_obstacle)
            
            if 0 < distance < self.influence_distance:
                # Repulsive force inversely proportional to distance
                force_magnitude = self.kr * (1.0/distance - 1.0/self.influence_distance) / (distance**2)
                repulsive_force += force_magnitude * (to_obstacle / distance)
        
        # Total force
        total_force = attractive_force + repulsive_force
        
        return total_force, attractive_force, repulsive_force
    
    def compute_velocity(self, current_position, current_orientation, obstacles, max_speed=0.5):
        """Compute desired velocity"""
        force, _, _ = self.compute_force(current_position, obstacles)
        
        # Desired direction
        if np.linalg.norm(force) > 0:
            desired_direction = force / np.linalg.norm(force)
        else:
            desired_direction = np.array([1.0, 0.0])
        
        # Current direction
        current_direction = np.array([
            np.cos(current_orientation),
            np.sin(current_orientation)
        ])
        
        # Desired velocity
        desired_velocity = desired_direction * max_speed
        
        # Angular velocity to align with desired direction
        cross = np.cross(current_direction, desired_direction)
        dot = np.dot(current_direction, desired_direction)
        angular_velocity = np.arctan2(cross, dot)
        
        return desired_velocity, angular_velocity


class NavigationController(LightweightRobotController):
    """
    Enhanced controller for autonomous navigation
    """
    
    def __init__(self, config=None, goal_position=None, avoid_obstacles=True):
        super().__init__(config)
        
        self.goal_position = np.array(goal_position or [2.0, 0.0])
        self.avoid_obstacles = avoid_obstacles
        
        # Navigation components
        self.obstacle_detector = ObstacleDetector()
        self.navigator = PotentialFieldNavigator(self.goal_position)
        
        # State
        self.state = NavigationState.MOVING_TO_GOAL
        self.estimated_position = np.array([0.0, 0.0])
        self.estimated_orientation = 0.0
        
        # Odometry (simple dead reckoning)
        self.last_update = time.time()
        
        # Recovery
        self.stuck_counter = 0
        self.last_position = self.estimated_position.copy()
        self.position_history = deque(maxlen=30)
        
        # Statistics
        self.distance_traveled = 0.0
        self.obstacles_avoided = 0
        self.recoveries = 0
    
    def update_odometry(self, action, dt):
        """Update position estimate (dead reckoning)"""
        # Update position based on action
        linear_vel = action.linear_velocity[0]
        angular_vel = action.angular_velocity[2]
        
        # Update orientation
        self.estimated_orientation += angular_vel * dt
        self.estimated_orientation = np.mod(self.estimated_orientation + np.pi, 2*np.pi) - np.pi
        
        # Update position
        dx = linear_vel * np.cos(self.estimated_orientation) * dt
        dy = linear_vel * np.sin(self.estimated_orientation) * dt
        
        self.estimated_position[0] += dx
        self.estimated_position[1] += dy
        
        # Update distance traveled
        self.distance_traveled += np.sqrt(dx**2 + dy**2)
        
        # Update history
        self.position_history.append(self.estimated_position.copy())
    
    def check_if_stuck(self):
        """Check if robot is stuck (not moving)"""
        if len(self.position_history) < 30:
            return False
        
        # Calculate variance in recent positions
        positions = np.array(list(self.position_history))
        variance = np.var(positions, axis=0)
        
        # If variance is very low, we're stuck
        if np.sum(variance) < 0.001:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        return self.stuck_counter > 10
    
    def distance_to_goal(self):
        """Calculate distance to goal"""
        return np.linalg.norm(self.goal_position - self.estimated_position)
    
    def compute_action(self, observation):
        """Compute action based on current state"""
        # Update timing
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Detect obstacles
        if self.avoid_obstacles:
            obstacle_density, clear_sector, mask = self.obstacle_detector.detect(observation.image)
        else:
            obstacle_density = {'left': 0, 'center': 0, 'right': 0}
            clear_sector = 'center'
            mask = None
        
        # Check distance to goal
        distance = self.distance_to_goal()
        
        if distance < 0.2:  # Reached goal (within 20cm)
            if self.state != NavigationState.REACHED_GOAL:
                print(f"✓ Goal reached! Distance traveled: {self.distance_traveled:.2f}m")
                self.state = NavigationState.REACHED_GOAL
        
        # Check if stuck
        if self.check_if_stuck() and self.state == NavigationState.MOVING_TO_GOAL:
            print("! Robot appears stuck, initiating recovery")
            self.state = NavigationState.RECOVERING
            self.recoveries += 1
        
        # State machine
        action = None
        
        if self.state == NavigationState.MOVING_TO_GOAL:
            action = self._move_to_goal(obstacle_density, clear_sector)
        
        elif self.state == NavigationState.AVOIDING_OBSTACLE:
            action = self._avoid_obstacle(obstacle_density, clear_sector)
        
        elif self.state == NavigationState.RECOVERING:
            action = self._recover()
        
        elif self.state == NavigationState.REACHED_GOAL:
            action = self._at_goal()
        
        elif self.state == NavigationState.STUCK:
            action = self._stuck()
        
        # Update odometry
        if action:
            self.update_odometry(action, dt)
        
        return action or Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _move_to_goal(self, obstacle_density, clear_sector):
        """Move toward goal"""
        # Check for obstacles
        if self.avoid_obstacles and obstacle_density['center'] > 0.3:
            print("! Obstacle detected, avoiding")
            self.state = NavigationState.AVOIDING_OBSTACLE
            self.obstacles_avoided += 1
            return self._avoid_obstacle(obstacle_density, clear_sector)
        
        # Navigate using potential fields
        # Simplified: estimate obstacle positions from density
        obstacles = []
        if obstacle_density['center'] > 0.1:
            # Obstacle ahead
            obstacles.append(self.estimated_position + np.array([0.5, 0.0]))
        
        desired_vel, angular_vel = self.navigator.compute_velocity(
            self.estimated_position,
            self.estimated_orientation,
            obstacles,
            max_speed=0.4
        )
        
        # Convert to robot frame
        linear_vel = desired_vel[0] * np.cos(self.estimated_orientation) + \
                     desired_vel[1] * np.sin(self.estimated_orientation)
        
        return Action(
            linear_velocity=np.array([linear_vel, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, angular_vel]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _avoid_obstacle(self, obstacle_density, clear_sector):
        """Avoid obstacle"""
        # Choose direction based on clear sector
        if clear_sector == 'left':
            angular_vel = 0.5  # Turn left
        elif clear_sector == 'right':
            angular_vel = -0.5  # Turn right
        else:
            angular_vel = 0.0
        
        # Move slowly while turning
        linear_vel = 0.1
        
        # If no obstacles ahead, return to goal-seeking
        if obstacle_density['center'] < 0.2:
            self.state = NavigationState.MOVING_TO_GOAL
        
        return Action(
            linear_velocity=np.array([linear_vel, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, angular_vel]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _recover(self):
        """Recovery behavior"""
        # Simple recovery: back up and turn
        recovery_time = time.time() - self.last_update
        
        if recovery_time < 1.0:
            # Back up
            return Action(
                linear_velocity=np.array([-0.2, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                joint_commands=np.zeros(6),
                gripper_state=0.0
            )
        elif recovery_time < 2.0:
            # Turn
            return Action(
                linear_velocity=np.array([0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.8]),
                joint_commands=np.zeros(6),
                gripper_state=0.0
            )
        else:
            # Resume normal operation
            self.state = NavigationState.MOVING_TO_GOAL
            self.stuck_counter = 0
            return self._move_to_goal({}, 'center')
    
    def _at_goal(self):
        """At goal - stop"""
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def _stuck(self):
        """Stuck - stop"""
        return Action(
            linear_velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            joint_commands=np.zeros(6),
            gripper_state=0.0
        )
    
    def compute_reward(self, observation, action):
        """Compute navigation reward"""
        # Reward for getting closer to goal
        distance = self.distance_to_goal()
        reward = -distance  # Negative distance (closer is better)
        
        # Bonus for reaching goal
        if distance < 0.2:
            reward += 10.0
        
        # Penalty for obstacles
        if self.avoid_obstacles:
            obstacle_density, _, _ = self.obstacle_detector.detect(observation.image)
            reward -= obstacle_density['center'] * 0.5
        
        # Penalty for being stuck
        if self.state == NavigationState.STUCK or self.state == NavigationState.RECOVERING:
            reward -= 1.0
        
        return reward


def draw_navigation_ui(image, controller, obstacle_mask):
    """Draw navigation UI"""
    h, w = image.shape[:2]
    result = image.copy()
    
    # Overlay obstacle mask
    if obstacle_mask is not None:
        # Expand mask to full image height
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[h//2:, :] = obstacle_mask
        
        mask_color = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 1] = 0  # Remove green
        mask_color[:, :, 2] = 0  # Remove red
        result = cv2.addWeighted(result, 0.7, mask_color, 0.3, 0)
    
    # Draw sectors
    sector_width = w // 3
    cv2.line(result, (sector_width, 0), (sector_width, h), (100, 100, 100), 1)
    cv2.line(result, (2*sector_width, 0), (2*sector_width, h), (100, 100, 100), 1)
    
    # Status overlay
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Title
    cv2.putText(result, "AUTONOMOUS NAVIGATION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # State
    state_colors = {
        NavigationState.MOVING_TO_GOAL: (0, 255, 0),
        NavigationState.AVOIDING_OBSTACLE: (255, 255, 0),
        NavigationState.RECOVERING: (255, 150, 0),
        NavigationState.REACHED_GOAL: (0, 255, 255),
        NavigationState.STUCK: (0, 0, 255)
    }
    
    color = state_colors.get(controller.state, (255, 255, 255))
    cv2.putText(result, f"State: {controller.state.name}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Position
    pos = controller.estimated_position
    cv2.putText(result, f"Position: ({pos[0]:.2f}, {pos[1]:.2f})", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Goal
    goal = controller.goal_position
    distance = controller.distance_to_goal()
    cv2.putText(result, f"Goal: ({goal[0]:.2f}, {goal[1]:.2f})", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(result, f"Distance: {distance:.2f}m", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Stats
    cv2.putText(result, f"Traveled: {controller.distance_traveled:.2f}m", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    cv2.putText(result, f"Obstacles avoided: {controller.obstacles_avoided}", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    # Draw goal indicator
    goal_screen_x = int(w/2 + (goal[1] - pos[1]) * 100)
    if 0 < goal_screen_x < w:
        cv2.arrowedLine(result, (w//2, h-50), (goal_screen_x, h-50), (0, 255, 0), 3)
        cv2.putText(result, "GOAL", (goal_screen_x - 20, h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result


def main():
    """Main function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Autonomous Navigation Example')
    parser.add_argument('--goal', nargs=2, type=float, default=[2.0, 0.0],
                       help='Goal position (x, y) in meters')
    parser.add_argument('--no-obstacles', action='store_true',
                       help='Disable obstacle avoidance')
    parser.add_argument('--save', type=str, default='navigation_controller',
                       help='Save path')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("AUTONOMOUS NAVIGATION")
    print("=" * 70)
    print()
    print(f"Goal position: ({args.goal[0]:.2f}, {args.goal[1]:.2f})")
    print(f"Obstacle avoidance: {not args.no_obstacles}")
    print()
    
    # Create controller
    config = {
        'control_frequency': 20,
        'learning_rate': 0.001
    }
    
    print("Initializing controller...")
    controller = NavigationController(
        config=config,
        goal_position=args.goal,
        avoid_obstacles=not args.no_obstacles
    )
    
    # Load if exists
    if Path(args.save).exists():
        try:
            controller.load(args.save)
            print(f"✓ Loaded: {len(controller.replay_buffer)} experiences")
        except:
            print("Starting fresh")
    
    print()
    print("Starting navigation...")
    controller.start()
    print("✓ Navigation started!")
    print()
    
    try:
        while True:
            # Get observation
            obs = controller.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue
            
            # Detect obstacles
            if controller.avoid_obstacles:
                _, _, mask = controller.obstacle_detector.detect(obs.image)
            else:
                mask = None
            
            # Display
            display_image = draw_navigation_ui(obs.image, controller, mask)
            cv2.imshow('Autonomous Navigation', display_image)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                # Reset
                controller.estimated_position = np.array([0.0, 0.0])
                controller.estimated_orientation = 0.0
                controller.distance_traveled = 0.0
                controller.state = NavigationState.MOVING_TO_GOAL
                print("Reset to start")
            
            # Check if reached goal
            if controller.state == NavigationState.REACHED_GOAL:
                time.sleep(2)
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
        print("NAVIGATION SUMMARY")
        print("=" * 70)
        print(f"Final position:      ({controller.estimated_position[0]:.2f}, {controller.estimated_position[1]:.2f})")
        print(f"Goal position:       ({controller.goal_position[0]:.2f}, {controller.goal_position[1]:.2f})")
        print(f"Final distance:      {controller.distance_to_goal():.2f}m")
        print(f"Distance traveled:   {controller.distance_traveled:.2f}m")
        print(f"Obstacles avoided:   {controller.obstacles_avoided}")
        print(f"Recovery attempts:   {controller.recoveries}")
        print(f"Final state:         {controller.state.name}")
        print()


if __name__ == '__main__':
    main()
