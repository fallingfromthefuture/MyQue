# Lightweight ML Robot Control - Quick Reference

## Installation (One-Time Setup)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev libatlas-base-dev libopenblas-dev

# Install Python packages
pip3 install numpy scipy pillow opencv-python tflite-runtime
```

## Basic Usage

```python
from lightweight_ml_robot_control import LightweightRobotController

# Create and start
controller = LightweightRobotController()
controller.start()

# Load previous training
controller.load('saved_controller')

# Stop and save
controller.stop()
controller.save('saved_controller')
```

## File Locations

- **Main Script:** `lightweight_ml_robot_control.py`
- **Documentation:** `ROBOT_CONTROL_DOCUMENTATION.md`
- **Examples:** `example_*.py`
- **Saved Models:** `saved_controller/`

## Key Classes

### LightweightRobotController
Main controller class

**Methods:**
- `start()` - Start control loop (30 Hz)
- `stop()` - Stop control loop
- `get_observation()` - Get camera + sensor data
- `compute_action(obs)` - Generate action
- `demonstrate(obs, action, reward)` - Add training example
- `save(path)` - Save state
- `load(path)` - Load state

### Action
Robot action command

```python
from lightweight_ml_robot_control import Action
import numpy as np

action = Action(
    linear_velocity=np.array([0.1, 0, 0]),  # m/s
    angular_velocity=np.array([0, 0, 0.5]),  # rad/s
    joint_commands=np.zeros(6),  # Joint angles
    gripper_state=0.5  # 0=open, 1=closed
)
```

### Observation
Sensor observation

```python
obs = controller.get_observation()
# obs.image - numpy array (H, W, 3)
# obs.robot_state - RobotState object
# obs.timestamp - float
```

## Common Tasks

### Teach by Demonstration

```python
# Get observation
obs = controller.get_observation()

# Define action (from joystick, keyboard, etc)
action = Action(...)

# Add to training
controller.demonstrate(obs, action, reward=1.0)
```

### Custom Hardware Interface

```python
class MyRobot(LightweightRobotController):
    def execute_action(self, action):
        # Your motor control code here
        self.left_motor.set_speed(action.linear_velocity[0])
        self.right_motor.set_speed(action.angular_velocity[2])
```

### Custom Sensors

```python
class MyRobot(LightweightRobotController):
    def get_observation(self):
        obs = super().get_observation()
        # Add your sensors
        obs.robot_state.sensor_data['distance'] = self.ultrasonic.read()
        return obs
```

### Custom Rewards

```python
class MyRobot(LightweightRobotController):
    def compute_reward(self, observation, action):
        # Your reward logic
        objects = self.object_detector.detect(observation.image)
        return 1.0 if len(objects) > 0 else 0.0
```

## Configuration Options

```python
config = {
    'vision_model_path': 'model.tflite',  # Optional TFLite model
    'control_frequency': 30,  # Hz (lower for slower CPUs)
    'learning_rate': 0.0001,  # Policy learning rate
}

controller = LightweightRobotController(config)
```

## Performance Tips

**For Raspberry Pi 3:**
- Set `control_frequency=20`
- Use camera resolution 320x240
- Reduce buffer size: `ExperienceReplayBuffer(max_size=5000)`

**For Raspberry Pi 4:**
- Keep default settings (30 Hz, 640x480)
- Can use `max_size=10000`

**For Jetson Nano:**
- Can increase to 60 Hz
- Use higher resolution (1280x720)
- Larger buffer (20000+)

## Running Examples

```bash
# Basic autonomous control
python3 example_basic_control.py

# Keyboard teleoperation (teach mode)
python3 example_teleoperation.py
```

## Troubleshooting

**No camera detected:**
```bash
ls /dev/video*  # Check USB cameras
vcgencmd get_camera  # Check CSI camera
```

**Out of memory:**
```python
# Reduce buffer size
replay_buffer = ExperienceReplayBuffer(max_size=1000)
# Reduce network size
policy = LightweightPolicyNetwork(..., hidden_dims=[32, 16])
```

**Slow performance:**
```python
# Lower camera resolution
controller.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
controller.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# Reduce control frequency
controller.control_frequency = 20
```

**TFLite not found:**
```bash
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

## File Structure

```
saved_controller/
├── policy.pkl           # Trained neural network
├── replay_buffer.pkl    # Experience history
└── config.json         # Configuration
```

## Learning Process

1. **Collection Phase** (0-32 experiences)
   - System collects observations and actions
   - No learning yet, just exploring

2. **Learning Phase** (32+ experiences)
   - Background learning starts (10 Hz)
   - Policy improves continuously
   - Keeps collecting more data

3. **Autonomous Phase** (1000+ experiences)
   - Robot performs tasks autonomously
   - Still learns from outcomes
   - Can be fine-tuned with demonstrations

## Key Metrics

- **Experiences:** Total observation-action pairs collected
- **Updates:** Number of learning iterations performed
- **Buffer Size:** Current replay memory usage
- **Control Frequency:** Control loop rate (Hz)

## Support

- **Documentation:** `ROBOT_CONTROL_DOCUMENTATION.md`
- **GitHub:** github.com/theroboos/lightweight-ml-control
- **Email:** support@theroboos.com

---

**Quick Start Checklist:**

- [ ] Install dependencies
- [ ] Connect camera
- [ ] Run `example_basic_control.py`
- [ ] Teach with `example_teleoperation.py`
- [ ] Customize for your hardware
- [ ] Deploy and iterate!

---

**Version:** 1.0.0 | **Last Updated:** December 2025
