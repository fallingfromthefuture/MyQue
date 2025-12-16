#!/usr/bin/env python3
"""
Teleoperation Example
Teach the robot by controlling it with keyboard
The robot learns from your demonstrations
"""

import sys
import time
import numpy as np
import threading
try:
    import cv2
except ImportError:
    print("OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)

from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action
)

class KeyboardTeleop:
    """Simple keyboard teleoperation"""
    
    def __init__(self):
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.gripper_state = 0.0
        self.running = True
        
    def get_action(self) -> Action:
        """Get current action from keyboard state"""
        return Action(
            linear_velocity=np.array([self.linear_velocity, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, self.angular_velocity]),
            joint_commands=np.zeros(6),
            gripper_state=self.gripper_state
        )
    
    def process_key(self, key):
        """Process keyboard input"""
        # Speed control
        speed = 0.3
        turn_speed = 1.0
        
        if key == ord('w') or key == ord('W'):
            self.linear_velocity = speed
        elif key == ord('s') or key == ord('S'):
            self.linear_velocity = -speed
        elif key == ord('a') or key == ord('A'):
            self.angular_velocity = turn_speed
        elif key == ord('d') or key == ord('D'):
            self.angular_velocity = -turn_speed
        elif key == ord('o') or key == ord('O'):
            self.gripper_state = 1.0  # Close gripper
        elif key == ord('p') or key == ord('P'):
            self.gripper_state = 0.0  # Open gripper
        elif key == ord(' '):
            # Space to stop
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
        elif key == 27:  # ESC
            self.running = False
            return False
        
        return True

def draw_controls(image, teleop, stats):
    """Draw control UI on image"""
    h, w = image.shape[:2]
    
    # Semi-transparent overlay
    overlay = image.copy()
    
    # Control panel background
    cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    # Title
    cv2.putText(image, "Robot Teleoperation - Learning Mode", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instructions
    y = 60
    instructions = [
        "W/S: Forward/Backward",
        "A/D: Turn Left/Right",
        "O/P: Close/Open Gripper",
        "SPACE: Stop",
        "ESC: Quit"
    ]
    
    for instruction in instructions:
        cv2.putText(image, instruction, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
    
    # Current state
    y = h - 120
    cv2.rectangle(overlay, (10, y-10), (w-10, h-10), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    cv2.putText(image, f"Linear Vel: {teleop.linear_velocity:+.2f} m/s", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv2.putText(image, f"Angular Vel: {teleop.angular_velocity:+.2f} rad/s", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv2.putText(image, f"Gripper: {'CLOSED' if teleop.gripper_state > 0.5 else 'OPEN'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Learning stats
    y += 30
    cv2.putText(image, f"Experiences: {stats['experiences']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y += 20
    cv2.putText(image, f"Learning Updates: {stats['updates']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return image

def main():
    print("=" * 60)
    print("Teleoperation Mode - Teach Your Robot")
    print("=" * 60)
    print()
    print("This mode allows you to control the robot with your keyboard.")
    print("The robot will learn from your demonstrations!")
    print()
    print("Controls:")
    print("  W/S - Move forward/backward")
    print("  A/D - Turn left/right")
    print("  O/P - Close/Open gripper")
    print("  SPACE - Stop all movement")
    print("  ESC - Quit")
    print()
    input("Press Enter to start...")
    
    # Create controller
    print("\nInitializing robot controller...")
    config = {
        'control_frequency': 20,
        'learning_rate': 0.001
    }
    controller = LightweightRobotController(config)
    
    # Load previous training
    try:
        controller.load('saved_controller')
        print("Loaded previous training data")
    except:
        print("Starting fresh (no previous data)")
    
    # Create teleoperation handler
    teleop = KeyboardTeleop()
    
    # Stats
    stats = {
        'experiences': 0,
        'updates': 0,
        'last_demo_time': time.time()
    }
    
    print("\nStarting control loop...")
    controller.start()
    
    print("\nWindow will open showing camera view with controls.")
    print("Use keyboard to control the robot!")
    print()
    
    try:
        while teleop.running:
            # Get observation
            obs = controller.get_observation()
            if obs is None:
                print("Warning: No camera image available")
                time.sleep(0.1)
                continue
            
            # Get current action from teleop
            action = teleop.get_action()
            
            # Add demonstration if controlling (not idle for >0.5s)
            current_time = time.time()
            is_active = (abs(teleop.linear_velocity) > 0.01 or 
                        abs(teleop.angular_velocity) > 0.01)
            
            if is_active:
                # Add demonstration
                controller.demonstrate(obs, action, reward=1.0)
                stats['last_demo_time'] = current_time
                stats['experiences'] = len(controller.replay_buffer)
                stats['updates'] = controller.learner.update_counter
            
            # Draw UI on image
            display_image = obs.image.copy()
            display_image = draw_controls(display_image, teleop, stats)
            
            # Show image
            cv2.imshow('Robot Teleoperation', display_image)
            
            # Process key
            key = cv2.waitKey(30)
            if key != -1:
                if not teleop.process_key(key):
                    break
            
            # Decay velocities if no recent input
            if current_time - stats['last_demo_time'] > 0.5:
                teleop.linear_velocity *= 0.8
                teleop.angular_velocity *= 0.8
                if abs(teleop.linear_velocity) < 0.01:
                    teleop.linear_velocity = 0.0
                if abs(teleop.angular_velocity) < 0.01:
                    teleop.angular_velocity = 0.0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\n" + "=" * 60)
        print("Stopping and saving...")
        print("=" * 60)
        
        # Stop controller
        controller.stop()
        
        # Save
        controller.save('saved_controller')
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print summary
        print("\nSummary:")
        print(f"  Total demonstrations collected: {len(controller.replay_buffer)}")
        print(f"  Learning updates performed: {controller.learner.update_counter}")
        print(f"  Model saved to: saved_controller/")
        print()
        print("Great! The robot has learned from your demonstrations.")
        print("You can now run the basic control script to see it perform")
        print("the tasks autonomously!")
        print()

if __name__ == '__main__':
    main()
