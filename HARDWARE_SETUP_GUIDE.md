# Hardware Setup Guide
## Raspberry Pi Robot with ML Control

---

## Table of Contents
1. [Bill of Materials](#bill-of-materials)
2. [Wiring Diagrams](#wiring-diagrams)
3. [Camera Setup](#camera-setup)
4. [Motor Driver Setup](#motor-driver-setup)
5. [Sensor Integration](#sensor-integration)
6. [Power Supply](#power-supply)
7. [Assembly Tips](#assembly-tips)

---

## Bill of Materials

### Essential Components

| Component | Specification | Approx. Cost | Notes |
|-----------|--------------|--------------|-------|
| **Raspberry Pi 4** | 4GB RAM | $55 | 2GB works but 4GB recommended |
| **MicroSD Card** | 32GB Class 10 | $10 | SanDisk or Samsung |
| **Power Supply** | 5V 3A USB-C | $10 | Official RPi power supply |
| **Camera** | RPi Camera v2 or USB | $25 | 8MP, 1080p |
| **Motor Driver** | L298N Dual H-Bridge | $5 | 2A per channel |
| **Motors** | DC Gear Motors | $10 | 6V, 200 RPM |
| **Wheels** | 65mm diameter | $5 | With motor mounts |
| **Chassis** | Acrylic or 3D printed | $15 | Or build your own |
| **Battery** | 7.4V 2200mAh LiPo | $20 | With XT60 connector |
| **Voltage Regulator** | 5V 3A Buck converter | $5 | For RPi from battery |

**Total Essential: ~$160**

### Optional Components

| Component | Purpose | Cost |
|-----------|---------|------|
| IMU (MPU6050) | Motion sensing | $5 |
| Ultrasonic (HC-SR04) | Distance measurement | $3 |
| Servo Motor | Gripper/camera pan | $5 |
| GPIO Expansion Board | Easy connections | $10 |
| Heat Sinks | Cooling | $5 |
| Small Fan | Active cooling | $5 |
| Breadboard & Jumpers | Prototyping | $10 |

---

## Wiring Diagrams

### Basic Robot Configuration

```
                    ┌─────────────────────┐
                    │   Raspberry Pi 4    │
                    │                     │
   Camera ──────────┤ CSI Port            │
                    │                     │
                    │ GPIO Header         │
                    └──────┬──────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼────────┐      ┌────────▼────────┐
     │  Motor Driver   │      │  5V Regulator   │
     │    (L298N)      │      │   (Buck Conv.)  │
     └────┬──────┬─────┘      └────────┬────────┘
          │      │                     │
     Left │      │ Right               │
     Motor│      │Motor         ┌──────▼──────┐
          │      │              │  Battery    │
          │      │              │  7.4V LiPo  │
          │      │              └─────────────┘
          ▼      ▼
       ●─────────●  Wheels
```

### Raspberry Pi GPIO Pinout (Key Pins)

```
     3V3  (1) (2)  5V
   GPIO2  (3) (4)  5V
   GPIO3  (5) (6)  GND
   GPIO4  (7) (8)  GPIO14
     GND  (9) (10) GPIO15
  GPIO17 (11) (12) GPIO18
  GPIO27 (13) (14) GND
  GPIO22 (15) (16) GPIO23
     3V3 (17) (18) GPIO24
  GPIO10 (19) (20) GND
   GPIO9 (21) (22) GPIO25
  GPIO11 (23) (24) GPIO8
     GND (25) (26) GPIO7
```

---

## Camera Setup

### Option 1: Raspberry Pi Camera Module (Recommended)

**Wiring:**
1. Power off Raspberry Pi
2. Locate CSI camera port (between HDMI and USB)
3. Pull up on black tab gently
4. Insert ribbon cable (blue side facing USB ports)
5. Push down black tab to secure

**Enable Camera:**
```bash
sudo raspi-config
# Select: Interface Options → Camera → Enable
sudo reboot
```

**Test Camera:**
```bash
raspistill -o test.jpg
libcamera-hello
```

### Option 2: USB Camera

**Wiring:**
- Simply plug into USB port

**Test:**
```bash
ls /dev/video*
# Should show /dev/video0
```

---

## Motor Driver Setup

### L298N Dual H-Bridge

**Pins:**
- `IN1, IN2` - Left motor control
- `IN3, IN4` - Right motor control
- `ENA, ENB` - Enable pins (PWM speed control)
- `+12V, GND` - Power input
- `+5V` - Output (can power RPi if using 7-12V input)

**Wiring to Raspberry Pi:**

```
L298N          →  Raspberry Pi GPIO
─────────────────────────────────────
ENA (PWM)      →  GPIO 18 (Pin 12)
IN1            →  GPIO 17 (Pin 11)
IN2            →  GPIO 27 (Pin 13)
IN3            →  GPIO 22 (Pin 15)
IN4            →  GPIO 23 (Pin 16)
ENB (PWM)      →  GPIO 13 (Pin 33)
GND            →  GND (Pin 6)
```

**Wiring to Motors:**
```
L298N          →  Motors
─────────────────────────
OUT1, OUT2     →  Left Motor
OUT3, OUT4     →  Right Motor
```

**Wiring to Battery:**
```
Battery 7.4V   →  L298N +12V, GND
```

**Python Control Code:**
```python
from gpiozero import Motor, PWMOutputDevice

# Define motors
left_motor = Motor(forward=17, backward=27, enable=18)
right_motor = Motor(forward=22, backward=23, enable=13)

# Move forward
left_motor.forward(0.5)   # 50% speed
right_motor.forward(0.5)

# Turn left
left_motor.backward(0.3)
right_motor.forward(0.5)

# Stop
left_motor.stop()
right_motor.stop()
```

---

## Sensor Integration

### IMU (MPU6050) - I2C

**Wiring:**
```
MPU6050        →  Raspberry Pi
─────────────────────────────
VCC            →  3.3V (Pin 1)
GND            →  GND (Pin 6)
SCL            →  GPIO 3 (Pin 5)
SDA            →  GPIO 2 (Pin 3)
```

**Enable I2C:**
```bash
sudo raspi-config
# Interface Options → I2C → Enable
sudo reboot
```

**Install Library:**
```bash
pip3 install mpu6050-raspberrypi
```

**Python Code:**
```python
from mpu6050 import mpu6050

sensor = mpu6050(0x68)

# Read accelerometer
accel = sensor.get_accel_data()
print(f"Accel: x={accel['x']:.2f}, y={accel['y']:.2f}, z={accel['z']:.2f}")

# Read gyroscope
gyro = sensor.get_gyro_data()
print(f"Gyro: x={gyro['x']:.2f}, y={gyro['y']:.2f}, z={gyro['z']:.2f}")
```

### Ultrasonic Sensor (HC-SR04)

**Wiring:**
```
HC-SR04        →  Raspberry Pi
─────────────────────────────
VCC            →  5V (Pin 2)
Trig           →  GPIO 24 (Pin 18)
Echo           →  GPIO 25 (Pin 22) + Voltage Divider*
GND            →  GND (Pin 6)

*Echo outputs 5V, RPi GPIO is 3.3V tolerant
Use voltage divider: Echo → 1kΩ → GPIO 25 → 2kΩ → GND
```

**Python Code:**
```python
from gpiozero import DistanceSensor

sensor = DistanceSensor(echo=25, trigger=24)

while True:
    distance = sensor.distance * 100  # Convert to cm
    print(f"Distance: {distance:.1f} cm")
    time.sleep(0.1)
```

### Servo Motor (for Gripper/Pan-Tilt)

**Wiring:**
```
Servo          →  Raspberry Pi
─────────────────────────────
Signal (Orange)→  GPIO 12 (Pin 32)
VCC (Red)      →  5V (Pin 4)
GND (Brown)    →  GND (Pin 6)
```

**Python Code:**
```python
from gpiozero import Servo

servo = Servo(12)

# Positions
servo.min()   # Minimum angle
servo.mid()   # Center
servo.max()   # Maximum angle

# Specific angle (-1 to 1)
servo.value = 0.5  # 75% toward max
```

---

## Power Supply

### Power Architecture

```
┌─────────────┐
│  7.4V LiPo  │
│   Battery   │
└──────┬──────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ Motor Driver │   │ Buck         │
│   (L298N)    │   │ Converter    │
│              │   │ 7.4V → 5V 3A │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
   [Motors]       [Raspberry Pi]
                   [Servo Motors]
                   [Sensors 5V]
```

### Battery Selection

**7.4V 2200mAh LiPo (Recommended):**
- Runtime: 1-2 hours
- Weight: 120g
- Discharge rate: 25C minimum

**Power Calculations:**
- Raspberry Pi 4: 2-3W (0.6A @ 5V)
- Motors (2x): 6W peak (1A @ 6V each)
- Camera: 0.5W
- **Total: ~10W peak**
- **Battery life: 2200mAh × 7.4V / 10W ≈ 1.6 hours**

### Buck Converter Setup

**Adjusting Output Voltage:**
1. Connect multimeter to output
2. Turn adjustment potentiometer
3. Set to exactly 5.0V (not higher!)
4. Test under load

**Connection:**
```
Battery +  →  Buck IN+
Battery -  →  Buck IN- and Motor Driver GND
Buck OUT+  →  RPi GPIO 5V (Pin 2 or 4)
Buck OUT-  →  RPi GND (Pin 6)
```

---

## Assembly Tips

### 1. Build Order

1. **Mount Raspberry Pi** to chassis
2. **Mount motor driver** near motors
3. **Mount battery holder** (center of chassis for balance)
4. **Connect motors** to driver
5. **Wire power** (battery → buck → RPi)
6. **Wire signals** (RPi GPIO → motor driver)
7. **Mount camera** at front
8. **Add sensors** as needed
9. **Cable management** with zip ties

### 2. Cooling

Raspberry Pi can get hot during ML inference.

**Passive Cooling:**
- Aluminum heat sinks on CPU and RAM
- Thermal paste or pads

**Active Cooling:**
- 30mm fan powered from 5V GPIO
- Mount above CPU

```python
# Fan control (optional)
from gpiozero import OutputDevice

fan = OutputDevice(14)  # GPIO 14

# Temperature check
def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        return float(f.read()) / 1000

# Control fan
if get_temp() > 70:
    fan.on()
else:
    fan.off()
```

### 3. Cable Management

- Use zip ties or velcro straps
- Keep power cables away from signal cables
- Use different colored wires (red=+, black=GND, etc.)
- Label connections with tape
- Leave some slack for movement

### 4. Vibration Dampening

- Use rubber standoffs for RPi mounting
- Secure all connectors with hot glue (optional)
- Mount camera with foam padding

---

## Testing Checklist

### Power System
- [ ] Battery voltage correct (7.4V ±0.5V)
- [ ] Buck converter output 5.0V ±0.1V
- [ ] RPi boots successfully
- [ ] No excessive heat

### Motors
- [ ] Left motor forward/backward
- [ ] Right motor forward/backward
- [ ] PWM speed control working
- [ ] No grinding or unusual sounds

### Camera
- [ ] Camera detected (`raspistill -o test.jpg`)
- [ ] Image quality good
- [ ] Frame rate acceptable (30 FPS)

### Sensors (if installed)
- [ ] IMU readings sensible
- [ ] Ultrasonic distance accurate
- [ ] Servo movement smooth

### Software
- [ ] Dependencies installed
- [ ] Control script runs
- [ ] Camera stream displays
- [ ] Actions execute correctly

---

## Common Issues

### RPi Won't Boot
- Check power supply (needs 3A)
- Check SD card (reformat and re-flash)
- Check voltage at GPIO pins (should be 5V)

### Motors Don't Move
- Check motor driver connections
- Verify GPIO pin numbers in code
- Check battery voltage (>7V)
- Test motor driver enable pins

### Camera Not Working
- Ensure camera is enabled in `raspi-config`
- Check ribbon cable connection
- Try `vcgencmd get_camera` (should show detected=1)

### Overheating
- Add heat sinks
- Add cooling fan
- Reduce CPU frequency if needed
- Ensure proper ventilation

---

## Safety Notes

⚠️ **Important Safety Information:**

1. **Never connect/disconnect while powered**
2. **Use correct voltage regulators** (RPi needs 5V, not 7.4V!)
3. **Add fuses** to battery connections (5A recommended)
4. **Monitor LiPo batteries** - never discharge below 3.0V per cell
5. **Keep batteries cool** - LiPo can catch fire if damaged
6. **Secure all connections** - loose wires can cause shorts
7. **Test on bench** before putting on robot

---

## Resources

### Datasheets
- [Raspberry Pi 4 Pinout](https://pinout.xyz)
- [L298N Motor Driver](https://www.st.com/resource/en/datasheet/l298.pdf)
- [MPU6050 IMU](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf)

### Tools Needed
- Soldering iron and solder
- Wire strippers
- Screwdrivers (Phillips and flat)
- Multimeter
- Helping hands/vice
- Heat shrink tubing

### Where to Buy
- **AdaFruit** - High quality components
- **SparkFun** - Great tutorials included
- **Amazon** - Quick shipping, kits available
- **AliExpress** - Budget option (longer shipping)

---

## Next Steps

After hardware is assembled:

1. **Test each component individually**
2. **Run `example_basic_control.py`** to verify integration
3. **Teach with `example_teleoperation.py`**
4. **Iterate on design** based on performance
5. **Add more sensors** as needed
6. **Share your build!** Post photos and videos

---

**Questions?** Check the main documentation or contact support@theroboos.com

**Share Your Build:** Post to #robot-builds on our Discord!

---

**Last Updated:** December 2025
