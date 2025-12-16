#!/usr/bin/env python3
"""
Lightweight State-of-the-Art ML Robot Control System
Optimized for Raspberry Pi and Edge Devices

This implementation uses:
- TensorFlow Lite for efficient inference
- MobileNetV3 for vision (optimized for mobile/edge)
- Online/Incremental learning for adaptation
- Sensor fusion with Kalman filtering
- Memory-efficient replay buffer
- Behavior cloning with fine-tuning
- Real-time object detection and tracking

Requirements:
    pip install tensorflow-lite numpy opencv-python pillow scipy

Hardware Support:
    - Raspberry Pi 3B+ / 4 / 5
    - Jetson Nano
    - Any ARM/x86 Linux SBC
    - HD Camera (USB/CSI)
    - IMU, Ultrasonic, or other sensors

Author: RoboOS AI Team
License: MIT
"""

import os
import json
import time
import threading
import pickle
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Core ML and Vision
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TFLite not available. Install with: pip install tflite-runtime")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RobotState:
    """Current robot state"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw] or quaternion
    joint_angles: np.ndarray  # Joint positions
    joint_velocities: np.ndarray  # Joint velocities
    sensor_data: Dict[str, Any]  # Additional sensor readings
    timestamp: float

@dataclass
class Action:
    """Robot action"""
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    joint_commands: np.ndarray  # Joint position/velocity commands
    gripper_state: float  # 0.0 (open) to 1.0 (closed)

@dataclass
class Observation:
    """Sensor observation"""
    image: np.ndarray  # Camera image
    depth: Optional[np.ndarray]  # Depth map if available
    robot_state: RobotState
    timestamp: float


# ============================================================================
# LIGHTWEIGHT NEURAL NETWORK MODELS
# ============================================================================

class MobileNetVisionEncoder:
    """
    Lightweight vision encoder using MobileNetV3
    Optimized for Raspberry Pi inference
    """
    
    def __init__(self, model_path: Optional[str] = None, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.interpreter = None
        
        if model_path and os.path.exists(model_path) and TFLITE_AVAILABLE:
            self.load_model(model_path)
        else:
            print("No model provided. Using feature extraction only.")
    
    def load_model(self, model_path: str):
        """Load TFLite model"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image"""
        if self.interpreter is None:
            # Fallback: simple feature extraction
            return self._extract_simple_features(image)
        
        # Preprocess
        img = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        features = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return features.flatten()
    
    def _extract_simple_features(self, image: np.ndarray) -> np.ndarray:
        """Simple feature extraction as fallback"""
        # Resize to small size
        small = cv2.resize(image, (32, 32))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Compute histograms
        hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_hue = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_sat = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Combine features
        features = np.concatenate([
            hist_gray.flatten() / 1000.0,
            hist_hue.flatten() / 1000.0,
            hist_sat.flatten() / 1000.0,
            [edge_density]
        ])
        
        return features


class LightweightPolicyNetwork:
    """
    Lightweight policy network for robot control
    Uses simple fully-connected layers optimized for edge devices
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights with He initialization
        self.weights = []
        self.biases = []
        
        layer_dims = [state_dim] + hidden_dims + [action_dim]
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros(layer_dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        x = state
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)  # ReLU
        
        # Output layer with tanh for bounded actions
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        x = np.tanh(x)
        
        return x
    
    def update(self, state: np.ndarray, target_action: np.ndarray, learning_rate: float = 0.001):
        """Simple gradient descent update (behavior cloning)"""
        # Forward pass
        activations = [state]
        x = state
        
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)
            activations.append(x)
        
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        output = np.tanh(x)
        activations.append(output)
        
        # Compute loss gradient (MSE)
        delta = 2 * (output - target_action)
        
        # Backward pass
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                # Output layer: gradient through tanh
                delta = delta * (1 - output**2)
            else:
                # Hidden layers: gradient through ReLU
                delta = delta * (activations[i+1] > 0)
            
            # Compute gradients
            grad_w = np.outer(activations[i], delta)
            grad_b = delta
            
            # Update weights
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # Propagate gradient
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
    
    def save(self, filepath: str):
        """Save model weights"""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load model weights"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        self.biases = data['biases']
        self.state_dim = data['state_dim']
        self.action_dim = data['action_dim']
        self.hidden_dims = data['hidden_dims']


# ============================================================================
# SENSOR PROCESSING
# ============================================================================

class SensorFusion:
    """
    Lightweight sensor fusion using complementary and Kalman filtering
    Combines camera, IMU, and other sensors
    """
    
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw
        
        # Kalman filter parameters
        self.P = np.eye(6) * 0.1  # State covariance
        self.Q = np.eye(6) * 0.01  # Process noise
        self.R = np.eye(3) * 0.1  # Measurement noise
        
        self.last_update = time.time()
    
    def predict(self, dt: float, acceleration: np.ndarray = None):
        """Prediction step"""
        if acceleration is None:
            acceleration = np.zeros(3)
        
        # State transition
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Update position and velocity
        state = np.concatenate([self.position, self.velocity])
        state = F @ state
        state[3:6] += acceleration * dt
        
        self.position = state[0:3]
        self.velocity = state[3:6]
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, acceleration: np.ndarray, gyroscope: np.ndarray):
        """Update with IMU data"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Update orientation from gyroscope
        self.orientation += gyroscope * dt
        
        # Normalize angles
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        # Predict with acceleration
        self.predict(dt, acceleration)
    
    def update_vision(self, position_measurement: np.ndarray):
        """Update with vision-based position estimate"""
        # Kalman update
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        
        y = position_measurement - self.position
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        state = np.concatenate([self.position, self.velocity])
        state = state + K @ y
        
        self.position = state[0:3]
        self.velocity = state[3:6]
        
        self.P = (np.eye(6) - K @ H) @ self.P
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current fused state"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy()
        }


class ObjectDetector:
    """
    Lightweight object detection using background subtraction and blob detection
    Can be replaced with TFLite model for better accuracy
    """
    
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 100
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
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
        
        return objects


# ============================================================================
# MEMORY AND LEARNING
# ============================================================================

class ExperienceReplayBuffer:
    """
    Memory-efficient experience replay buffer
    Stores observations, actions, and rewards for learning
    """
    
    def __init__(self, max_size: int = 10000, compress: bool = True):
        self.max_size = max_size
        self.compress = compress
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
    
    def add(self, observation: Observation, action: Action, reward: float, 
            next_observation: Observation, done: bool, priority: float = 1.0):
        """Add experience to buffer"""
        
        if self.compress:
            # Compress images to save memory
            obs_data = {
                'image': cv2.imencode('.jpg', observation.image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1],
                'robot_state': observation.robot_state,
                'timestamp': observation.timestamp
            }
            next_obs_data = {
                'image': cv2.imencode('.jpg', next_observation.image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1],
                'robot_state': next_observation.robot_state,
                'timestamp': next_observation.timestamp
            }
        else:
            obs_data = observation
            next_obs_data = next_observation
        
        experience = {
            'observation': obs_data,
            'action': action,
            'reward': reward,
            'next_observation': next_obs_data,
            'done': done
        }
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Dict]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        # Decompress images if needed
        if self.compress:
            for exp in batch:
                exp['observation']['image'] = cv2.imdecode(exp['observation']['image'], cv2.IMREAD_COLOR)
                exp['next_observation']['image'] = cv2.imdecode(exp['next_observation']['image'], cv2.IMREAD_COLOR)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'max_size': self.max_size,
                'compress': self.compress
            }, f)
    
    def load(self, filepath: str):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.buffer = deque(data['buffer'], maxlen=data['max_size'])
        self.priorities = deque(data['priorities'], maxlen=data['max_size'])
        self.max_size = data['max_size']
        self.compress = data['compress']


class OnlineLearner:
    """
    Online learning module for continuous adaptation
    Implements incremental learning and domain adaptation
    """
    
    def __init__(self, policy: LightweightPolicyNetwork, 
                 replay_buffer: ExperienceReplayBuffer,
                 learning_rate: float = 0.0001):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        
        self.update_counter = 0
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background learning"""
        self.running = True
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background learning"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _learning_loop(self):
        """Background learning loop"""
        while self.running:
            if len(self.replay_buffer) >= 32:
                self.update(batch_size=32)
            time.sleep(0.1)  # Update at 10Hz
    
    def update(self, batch_size: int = 32):
        """Perform one learning update"""
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Extract states and actions
        for exp in batch:
            # Get observation features
            obs = exp['observation']
            action = exp['action']
            
            # Simplified state (in practice, combine vision features + robot state)
            state = np.concatenate([
                obs['robot_state'].position,
                obs['robot_state'].orientation,
                obs['robot_state'].joint_angles
            ])
            
            # Target action
            target_action = np.concatenate([
                action.linear_velocity,
                action.angular_velocity,
                action.joint_commands,
                [action.gripper_state]
            ])
            
            # Update policy
            self.policy.update(state, target_action, self.learning_rate)
        
        self.update_counter += 1
        
        if self.update_counter % 100 == 0:
            print(f"Online learning update #{self.update_counter}")


# ============================================================================
# MAIN ROBOT CONTROLLER
# ============================================================================

class LightweightRobotController:
    """
    Main robot controller integrating vision, learning, and control
    Optimized for Raspberry Pi and edge devices
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.vision_encoder = MobileNetVisionEncoder(
            model_path=self.config.get('vision_model_path'),
            input_size=(224, 224)
        )
        
        self.object_detector = ObjectDetector()
        self.sensor_fusion = SensorFusion()
        
        # Policy network dimensions
        state_dim = 128 + 3 + 3 + 6  # vision features + pos + orient + joints
        action_dim = 3 + 3 + 6 + 1  # linear vel + angular vel + joints + gripper
        
        self.policy = LightweightPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 64]
        )
        
        self.replay_buffer = ExperienceReplayBuffer(max_size=10000, compress=True)
        self.learner = OnlineLearner(self.policy, self.replay_buffer)
        
        # Camera
        self.camera = None
        if CV2_AVAILABLE:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # State
        self.current_observation = None
        self.last_action = None
        
        # Control loop
        self.running = False
        self.control_thread = None
        self.control_frequency = 30  # Hz
        
        print("Lightweight Robot Controller initialized")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    def get_observation(self) -> Optional[Observation]:
        """Capture current observation from sensors"""
        if self.camera is None:
            return None
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Get robot state (in practice, read from hardware)
        robot_state = RobotState(
            position=self.sensor_fusion.position,
            orientation=self.sensor_fusion.orientation,
            joint_angles=np.zeros(6),  # Placeholder
            joint_velocities=np.zeros(6),  # Placeholder
            sensor_data={},
            timestamp=time.time()
        )
        
        observation = Observation(
            image=frame,
            depth=None,
            robot_state=robot_state,
            timestamp=time.time()
        )
        
        return observation
    
    def compute_action(self, observation: Observation) -> Action:
        """Compute action from observation using policy"""
        # Extract visual features
        visual_features = self.vision_encoder.encode(observation.image)
        
        # Pad or truncate to 128 dimensions
        if len(visual_features) < 128:
            visual_features = np.pad(visual_features, (0, 128 - len(visual_features)))
        else:
            visual_features = visual_features[:128]
        
        # Combine with robot state
        state = np.concatenate([
            visual_features,
            observation.robot_state.position,
            observation.robot_state.orientation,
            observation.robot_state.joint_angles
        ])
        
        # Get action from policy
        action_vector = self.policy.forward(state)
        
        # Parse action
        action = Action(
            linear_velocity=action_vector[0:3] * 0.5,  # Scale to reasonable values
            angular_velocity=action_vector[3:6] * 1.0,
            joint_commands=action_vector[6:12] * np.pi,
            gripper_state=(action_vector[12] + 1) / 2  # Map from [-1, 1] to [0, 1]
        )
        
        return action
    
    def execute_action(self, action: Action):
        """Execute action on robot hardware"""
        # In practice, send commands to motors/servos
        # This is a placeholder - implement hardware interface
        print(f"Executing action: linear_vel={action.linear_velocity[:2]}, "
              f"angular_vel={action.angular_velocity[2]:.2f}, "
              f"gripper={action.gripper_state:.2f}")
    
    def demonstrate(self, observation: Observation, action: Action, reward: float = 1.0):
        """Add demonstration to learning buffer"""
        next_observation = self.get_observation()
        if next_observation is None:
            return
        
        self.replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=False
        )
    
    def control_loop(self):
        """Main control loop"""
        dt = 1.0 / self.control_frequency
        
        while self.running:
            loop_start = time.time()
            
            # Get observation
            observation = self.get_observation()
            if observation is None:
                time.sleep(dt)
                continue
            
            # Compute action
            action = self.compute_action(observation)
            
            # Execute action
            self.execute_action(action)
            
            # Store for learning
            if self.last_action is not None:
                reward = self.compute_reward(observation, action)
                self.replay_buffer.add(
                    observation=self.current_observation,
                    action=self.last_action,
                    reward=reward,
                    next_observation=observation,
                    done=False
                )
            
            self.current_observation = observation
            self.last_action = action
            
            # Sleep to maintain frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def compute_reward(self, observation: Observation, action: Action) -> float:
        """Compute reward for learning (task-specific)"""
        # Example: reward smooth motions
        velocity_penalty = -0.1 * np.linalg.norm(action.linear_velocity)
        
        # Detect objects and reward proximity (example task)
        objects = self.object_detector.detect(observation.image)
        object_reward = 1.0 if len(objects) > 0 else 0.0
        
        return object_reward + velocity_penalty
    
    def start(self):
        """Start robot controller"""
        print("Starting robot controller...")
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
        # Start online learning
        self.learner.start()
        
        print("Robot controller started")
    
    def stop(self):
        """Stop robot controller"""
        print("Stopping robot controller...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join()
        
        self.learner.stop()
        
        if self.camera:
            self.camera.release()
        
        print("Robot controller stopped")
    
    def save(self, directory: str):
        """Save controller state"""
        os.makedirs(directory, exist_ok=True)
        
        # Save policy
        self.policy.save(os.path.join(directory, 'policy.pkl'))
        
        # Save replay buffer
        self.replay_buffer.save(os.path.join(directory, 'replay_buffer.pkl'))
        
        # Save config
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Controller saved to {directory}")
    
    def load(self, directory: str):
        """Load controller state"""
        # Load policy
        policy_path = os.path.join(directory, 'policy.pkl')
        if os.path.exists(policy_path):
            self.policy.load(policy_path)
            print("Policy loaded")
        
        # Load replay buffer
        buffer_path = os.path.join(directory, 'replay_buffer.pkl')
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)
            print(f"Replay buffer loaded ({len(self.replay_buffer)} experiences)")
        
        # Load config
        config_path = os.path.join(directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("Config loaded")


# ============================================================================
# UTILITIES
# ============================================================================

def create_mobilenet_tflite_model():
    """
    Helper function to create and export MobileNetV3 TFLite model
    Run this on a desktop/server, then transfer to Raspberry Pi
    """
    try:
        import tensorflow as tf
        
        # Load MobileNetV3
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Create model
        model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        with open('mobilenet_v3_small.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("MobileNetV3 TFLite model created: mobilenet_v3_small.tflite")
        print("Transfer this file to your Raspberry Pi")
        
    except ImportError:
        print("TensorFlow not available. Install with: pip install tensorflow")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage"""
    
    # Configuration
    config = {
        'vision_model_path': 'mobilenet_v3_small.tflite',  # Optional
        'control_frequency': 30,
        'learning_rate': 0.0001
    }
    
    # Create controller
    controller = LightweightRobotController(config)
    
    # Load previous training if available
    if os.path.exists('saved_controller'):
        controller.load('saved_controller')
    
    try:
        # Start controller
        controller.start()
        
        print("\nRobot controller running...")
        print("Press Ctrl+C to stop")
        
        # Run for demonstration
        while True:
            time.sleep(1)
            
            # Show stats
            buffer_size = len(controller.replay_buffer)
            update_count = controller.learner.update_counter
            print(f"Buffer: {buffer_size} experiences, Updates: {update_count}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        # Stop controller
        controller.stop()
        
        # Save state
        controller.save('saved_controller')
        
        print("Controller state saved")


if __name__ == '__main__':
    print("=" * 70)
    print("Lightweight ML Robot Control System")
    print("Optimized for Raspberry Pi and Edge Devices")
    print("=" * 70)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"TFLite available: {TFLITE_AVAILABLE}")
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print()
    
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available. Install with:")
        print("  pip install opencv-python")
        print()
    
    # Run main
    main()
