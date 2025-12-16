#!/usr/bin/env python3
"""
Hardware Interface Example
==========================
Complete example of integrating the ML controller with real hardware.
Demonstrates GPIO control, motor drivers, sensors, and servos.

Supports:
- L298N motor driver (differential drive)
- HC-SR04 ultrasonic sensor
- MPU6050 IMU
- Servo motors (for gripper/camera)
- Wheel encoders (optional)

Hardware Setup:
    Motors: GPIO 17, 18, 27, 22, 23, 13
    Ultrasonic: Trigger=24, Echo=25
    IMU: I2C (SDA=2, SCL=3)
    Servo: GPIO 12

Usage:
    python3 hardware_interface.py [--test] [--calibrate]

Author: RoboOS AI Team
License: MIT
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np

# GPIO library
try:
    from gpiozero import Motor, DistanceSensor, Servo, OutputDevice
    import gpiozero
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("Warning: gpiozero not available. Install with: pip install gpiozero")
    print("Running in simulation mode...")

# IMU library
try:
    from mpu6050 import mpu6050
    HAS_IMU = True
except ImportError:
    HAS_IMU = False
    print("Warning: mpu6050 library not available. Install with: pip install mpu6050-raspberrypi")

# Import the main controller
sys.path.insert(0, str(Path(__file__).parent.parent))
from lightweight_ml_robot_control import (
    LightweightRobotController,
    Action,
    RobotState
)


class HardwareInterface:
    """
    Hardware abstraction layer for robot control
    """
    
    def __init__(self, simulation=False):
        self.simulation = simulation or not HAS_GPIO
        
        if self.simulation:
            print("⚠ Running in SIMULATION mode (no actual hardware control)")
            self._init_simulation()
        else:
            print("✓ Initializing hardware...")
            self._init_hardware()
    
    def _init_simulation(self):
        """Initialize simulation mode"""
        # Simulated state
        self.sim_position = np.array([0.0, 0.0, 0.0])
        self.sim_velocity = np.array([0.0, 0.0, 0.0])
        self.sim_orientation = np.array([0.0, 0.0, 0.0])
        self.sim_distance = 1.0  # Simulated ultrasonic distance
        self.sim_imu_accel = np.array([0.0, 0.0, 9.81])
        self.sim_imu_gyro = np.array([0.0, 0.0, 0.0])
        
        print("  Simulation initialized")
    
    def _init_hardware(self):
        """Initialize real hardware"""
        # Motors (L298N driver)
        try:
            self.left_motor = Motor(
                forward=17,      # IN1
                backward=27,     # IN2
                enable=18        # ENA
            )
            self.right_motor = Motor(
                forward=22,      # IN3
                backward=23,     # IN4
                enable=13        # ENB
            )
            print("  ✓ Motors initialized")
        except Exception as e:
            print(f"  ✗ Motor init failed: {e}")
            self.left_motor = None
            self.right_motor = None
        
        # Ultrasonic sensor
        try:
            self.ultrasonic = DistanceSensor(
                echo=25,
                trigger=24,
                max_distance=3.0  # Max 3 meters
            )
            print("  ✓ Ultrasonic sensor initialized")
        except Exception as e:
            print(f"  ✗ Ultrasonic init failed: {e}")
            self.ultrasonic = None
        
        # IMU
        if HAS_IMU:
            try:
                self.imu = mpu6050(0x68)
                print("  ✓ IMU initialized")
            except Exception as e:
                print(f"  ✗ IMU init failed: {e}")
                self.imu = None
        else:
            self.imu = None
        
        # Servo (for gripper or camera)
        try:
            self.servo = Servo(12)
            self.servo.mid()  # Center position
            print("  ✓ Servo initialized")
        except Exception as e:
            print(f"  ✗ Servo init failed: {e}")
            self.servo = None
        
        # Status LED (optional)
        try:
            self.status_led = OutputDevice(14)
            self.status_led.off()
            print("  ✓ Status LED initialized")
        except Exception as e:
            self.status_led = None
    
    def set_motor_speeds(self, left_speed, right_speed):
        """
        Set motor speeds
        
        Args:
            left_speed: -1.0 to 1.0
            right_speed: -1.0 to 1.0
        """
        if self.simulation:
            # Update simulated velocity
            self.sim_velocity[0] = (left_speed + right_speed) / 2.0 * 0.5  # m/s
            angular = (right_speed - left_speed) * 0.5
            self.sim_velocity[2] = angular
            return
        
        if not self.left_motor or not self.right_motor:
            return
        
        # Left motor
        if left_speed > 0:
            self.left_motor.forward(abs(left_speed))
        elif left_speed < 0:
            self.left_motor.backward(abs(left_speed))
        else:
            self.left_motor.stop()
        
        # Right motor
        if right_speed > 0:
            self.right_motor.forward(abs(right_speed))
        elif right_speed < 0:
            self.right_motor.backward(abs(right_speed))
        else:
            self.right_motor.stop()
    
    def stop_motors(self):
        """Stop all motors"""
        if self.simulation:
            self.sim_velocity = np.array([0.0, 0.0, 0.0])
            return
        
        if self.left_motor:
            self.left_motor.stop()
        if self.right_motor:
            self.right_motor.stop()
    
    def get_distance(self):
        """Get distance from ultrasonic sensor (meters)"""
        if self.simulation:
            # Simulate obstacle
            return self.sim_distance
        
        if not self.ultrasonic:
            return None
        
        try:
            distance = self.ultrasonic.distance
            return distance if distance < 3.0 else 3.0
        except Exception as e:
            print(f"Ultrasonic error: {e}")
            return None
    
    def get_imu_data(self):
        """Get IMU acceleration and gyroscope data"""
        if self.simulation:
            return {
                'accel': self.sim_imu_accel,
                'gyro': self.sim_imu_gyro
            }
        
        if not self.imu:
            return None
        
        try:
            accel_data = self.imu.get_accel_data()
            gyro_data = self.imu.get_gyro_data()
            
            return {
                'accel': np.array([accel_data['x'], accel_data['y'], accel_data['z']]),
                'gyro': np.array([gyro_data['x'], gyro_data['y'], gyro_data['z']])
            }
        except Exception as e:
            print(f"IMU error: {e}")
            return None
    
    def set_servo_position(self, position):
        """
        Set servo position
        
        Args:
            position: 0.0 (min) to 1.0 (max)
        """
        if self.simulation:
            return
        
        if not self.servo:
            return
        
        try:
            # Convert 0-1 to servo range (-1 to 1)
            servo_value = position * 2.0 - 1.0
            self.servo.value = servo_value
        except Exception as e:
            print(f"Servo error: {e}")
    
    def set_led(self, state):
        """Set status LED"""
        if self.simulation:
            return
        
        if self.status_led:
            if state:
                self.status_led.on()
            else:
                self.status_led.off()
    
    def cleanup(self):
        """Cleanup GPIO"""
        if self.simulation:
            return
        
        print("Cleaning up hardware...")
        self.stop_motors()
        
        if self.servo:
            self.servo.mid()
        
        if self.status_led:
            self.status_led.off()
        
        # Close all devices
        if self.left_motor:
            self.left_motor.close()
        if self.right_motor:
            self.right_motor.close()
        if self.ultrasonic:
            self.ultrasonic.close()
        if self.servo:
            self.servo.close()
        if self.status_led:
            self.status_led.close()
        
        print("✓ Hardware cleanup complete")


class HardwareRobotController(LightweightRobotController):
    """
    Robot controller with hardware integration
    """
    
    def __init__(self, config=None, simulation=False):
        super().__init__(config)
        
        # Hardware interface
        self.hardware = HardwareInterface(simulation=simulation)
        
        # Wheel parameters (for differential drive)
        self.wheel_base = 0.15  # Distance between wheels (meters)
        self.wheel_radius = 0.033  # Wheel radius (meters)
        
        # Safety limits
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.5  # rad/s
        self.min_obstacle_distance = 0.2  # meters
        
        # Emergency stop
        self.emergency_stop_triggered = False
    
    def get_observation(self):
        """Get observation with hardware sensor data"""
        obs = super().get_observation()
        
        if obs is None:
            return None
        
        # Add ultrasonic distance
        distance = self.hardware.get_distance()
        if distance is not None:
            obs.robot_state.sensor_data['ultrasonic_distance'] = distance
        
        # Add IMU data
        imu_data = self.hardware.get_imu_data()
        if imu_data is not None:
            obs.robot_state.sensor_data['imu_accel'] = imu_data['accel']
            obs.robot_state.sensor_data['imu_gyro'] = imu_data['gyro']
            
            # Update sensor fusion with IMU
            self.sensor_fusion.update_imu(
                acceleration=imu_data['accel'],
                gyroscope=imu_data['gyro']
            )
        
        return obs
    
    def execute_action(self, action):
        """Execute action on hardware"""
        # Check emergency stop
        distance = self.hardware.get_distance()
        if distance is not None and distance < self.min_obstacle_distance:
            if not self.emergency_stop_triggered:
                print("⚠ EMERGENCY STOP: Obstacle too close!")
                self.emergency_stop_triggered = True
                self.hardware.set_led(True)
            
            # Override action to stop
            action = Action(
                linear_velocity=np.array([0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                joint_commands=np.zeros(6),
                gripper_state=action.gripper_state
            )
        else:
            if self.emergency_stop_triggered:
                print("✓ Emergency stop cleared")
                self.emergency_stop_triggered = False
                self.hardware.set_led(False)
        
        # Apply safety limits
        linear_vel = np.clip(action.linear_velocity[0], 
                            -self.max_linear_speed, 
                            self.max_linear_speed)
        angular_vel = np.clip(action.angular_velocity[2],
                             -self.max_angular_speed,
                             self.max_angular_speed)
        
        # Convert to differential drive
        # v = (vl + vr) / 2
        # w = (vr - vl) / L
        # vl = v - wL/2
        # vr = v + wL/2
        
        vl = linear_vel - (angular_vel * self.wheel_base / 2.0)
        vr = linear_vel + (angular_vel * self.wheel_base / 2.0)
        
        # Normalize to [-1, 1] for motor speeds
        max_wheel_speed = self.max_linear_speed
        left_motor_speed = vl / max_wheel_speed
        right_motor_speed = vr / max_wheel_speed
        
        # Clip to valid range
        left_motor_speed = np.clip(left_motor_speed, -1.0, 1.0)
        right_motor_speed = np.clip(right_motor_speed, -1.0, 1.0)
        
        # Send to motors
        self.hardware.set_motor_speeds(left_motor_speed, right_motor_speed)
        
        # Control gripper (servo)
        self.hardware.set_servo_position(action.gripper_state)
        
        # Call parent for logging
        super().execute_action(action)
    
    def compute_reward(self, observation, action):
        """Compute reward with hardware sensors"""
        reward = super().compute_reward(observation, action)
        
        # Bonus for avoiding obstacles
        distance = observation.robot_state.sensor_data.get('ultrasonic_distance')
        if distance is not None:
            if distance > 0.5:
                reward += 0.5  # Good clearance
            elif distance < 0.3:
                reward -= 1.0  # Too close
        
        # Penalty for emergency stop
        if self.emergency_stop_triggered:
            reward -= 2.0
        
        return reward
    
    def cleanup(self):
        """Cleanup hardware"""
        self.hardware.cleanup()


def test_hardware(hardware):
    """Test hardware components"""
    print("\n" + "=" * 70)
    print("HARDWARE TEST")
    print("=" * 70)
    
    # Test motors
    print("\n1. Testing motors...")
    print("  Forward...")
    hardware.set_motor_speeds(0.3, 0.3)
    time.sleep(2)
    
    print("  Backward...")
    hardware.set_motor_speeds(-0.3, -0.3)
    time.sleep(2)
    
    print("  Turn left...")
    hardware.set_motor_speeds(-0.3, 0.3)
    time.sleep(2)
    
    print("  Turn right...")
    hardware.set_motor_speeds(0.3, -0.3)
    time.sleep(2)
    
    print("  Stop")
    hardware.stop_motors()
    
    # Test ultrasonic
    print("\n2. Testing ultrasonic sensor...")
    for i in range(5):
        distance = hardware.get_distance()
        if distance is not None:
            print(f"  Distance: {distance:.2f}m")
        else:
            print("  No reading")
        time.sleep(0.5)
    
    # Test IMU
    print("\n3. Testing IMU...")
    for i in range(5):
        imu_data = hardware.get_imu_data()
        if imu_data is not None:
            accel = imu_data['accel']
            gyro = imu_data['gyro']
            print(f"  Accel: ({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}) m/s²")
            print(f"  Gyro:  ({gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f}) °/s")
        else:
            print("  No reading")
        time.sleep(0.5)
    
    # Test servo
    print("\n4. Testing servo...")
    print("  Min position")
    hardware.set_servo_position(0.0)
    time.sleep(1)
    
    print("  Mid position")
    hardware.set_servo_position(0.5)
    time.sleep(1)
    
    print("  Max position")
    hardware.set_servo_position(1.0)
    time.sleep(1)
    
    print("  Mid position")
    hardware.set_servo_position(0.5)
    
    # Test LED
    print("\n5. Testing LED...")
    for i in range(5):
        hardware.set_led(True)
        time.sleep(0.2)
        hardware.set_led(False)
        time.sleep(0.2)
    
    print("\n✓ Hardware test complete")


def main():
    """Main function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Hardware Interface Example')
    parser.add_argument('--test', action='store_true',
                       help='Run hardware test')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode (no hardware)')
    parser.add_argument('--save', type=str, default='hardware_controller',
                       help='Save path')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("HARDWARE INTERFACE - ROBOT CONTROL")
    print("=" * 70)
    print()
    
    if args.simulation or not HAS_GPIO:
        print("⚠ SIMULATION MODE")
        print()
    
    # Test mode
    if args.test:
        hardware = HardwareInterface(simulation=args.simulation)
        try:
            test_hardware(hardware)
        finally:
            hardware.cleanup()
        return 0
    
    # Create controller
    config = {
        'control_frequency': 20,
        'learning_rate': 0.001
    }
    
    print("Initializing robot controller...")
    controller = HardwareRobotController(
        config=config,
        simulation=args.simulation
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
    print("Robot is now running with hardware interface.")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    try:
        last_print = time.time()
        
        while True:
            current_time = time.time()
            
            # Print stats every second
            if current_time - last_print >= 1.0:
                obs = controller.current_observation
                if obs:
                    print(f"[{time.strftime('%H:%M:%S')}] ", end='')
                    
                    # Ultrasonic
                    distance = obs.robot_state.sensor_data.get('ultrasonic_distance')
                    if distance:
                        print(f"Distance: {distance:.2f}m | ", end='')
                    
                    # IMU
                    accel = obs.robot_state.sensor_data.get('imu_accel')
                    if accel is not None:
                        print(f"Accel: {np.linalg.norm(accel):.2f} | ", end='')
                    
                    # Learning
                    print(f"Exp: {len(controller.replay_buffer)} | ", end='')
                    print(f"Updates: {controller.learner.update_counter}")
                
                last_print = current_time
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        print("\n" + "=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        
        # Stop
        print("\nStopping robot...")
        controller.stop()
        
        # Cleanup hardware
        controller.cleanup()
        
        # Save
        print(f"\nSaving to: {args.save}")
        controller.save(args.save)
        
        print("\n✓ Shutdown complete")


if __name__ == '__main__':
    sys.exit(main() or 0)
