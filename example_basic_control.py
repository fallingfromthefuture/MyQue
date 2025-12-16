#!/usr/bin/env python3
"""
Basic Robot Control Example
Demonstrates simple usage of the lightweight ML robot controller
"""

import sys
import time
import numpy as np
from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action,
    RobotState
)

def main():
    print("=" * 60)
    print("Basic Robot Control Example")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        'control_frequency': 20,  # 20 Hz for this example
        'learning_rate': 0.001
    }
    
    # Create controller
    print("Initializing controller...")
    controller = LightweightRobotController(config)
    
    # Try to load previous training
    try:
        controller.load('saved_controller')
        print("Loaded previous training data")
    except:
        print("Starting fresh (no previous data)")
    
    print()
    
    try:
        # Start the controller
        print("Starting control loop...")
        controller.start()
        
        print("\nRobot is now running!")
        print("The system will:")
        print("  - Capture camera images")
        print("  - Process visual features")
        print("  - Generate and execute actions")
        print("  - Learn from experience continuously")
        print()
        print("Stats will update every second...")
        print("Press Ctrl+C to stop")
        print()
        
        # Monitor loop
        last_update = time.time()
        while True:
            current_time = time.time()
            
            # Update stats every second
            if current_time - last_update >= 1.0:
                buffer_size = len(controller.replay_buffer)
                updates = controller.learner.update_counter
                
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Experiences: {buffer_size:5d} | "
                      f"Learning updates: {updates:5d} | "
                      f"Status: {'Learning' if buffer_size >= 32 else 'Collecting'}")
                
                last_update = current_time
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopping robot...")
        print("=" * 60)
    
    finally:
        # Stop the controller
        controller.stop()
        
        # Save the trained model
        print("\nSaving controller state...")
        controller.save('saved_controller')
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total experiences collected: {len(controller.replay_buffer)}")
        print(f"Total learning updates: {controller.learner.update_counter}")
        print(f"Controller saved to: saved_controller/")
        print()
        print("Next time you run this, the robot will continue learning")
        print("from where it left off!")
        print()

if __name__ == '__main__':
    main()
