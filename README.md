# Bimanual Teleoperation of Ned 2 Robots using MediaPipe

A vision-based bimanual teleoperation system that enables intuitive control of dual Niryo Ned2 robotic arms using hand gestures and MediaPipe framework. This project addresses key limitations in traditional teleoperation methods through innovative gesture-based solutions.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Gesture Commands](#gesture-commands)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

This project implements a cost-effective vision-based teleoperation system for controlling dual Niryo Ned2 robotic arms using only a standard webcam and MediaPipe hand tracking. The system addresses four critical limitations in traditional teleoperation:

1. **Clutch Control**: Peace sign gesture for global pause/resume
2. **Latency Reduction**: Optimized motion smoothing and command intervals
3. **Gripper Interference**: Mode switching between arm control and gripper operation
4. **Independent Arm Control**: Selective pause mechanism for individual arms
5. **Emergency Stop**: Spacebar emergency stop for both arms

## Key Features

- **Vision-Based Control**: Uses MediaPipe for real-time hand tracking (no wearable sensors required)
- **Bimanual Coordination**: Simultaneous control of two robotic arms
- **Gesture Recognition**: 5 distinct hand gestures for different control modes
- **State Machine Architecture**: Robust control flow with comprehensive tutorial system
- **Cost Effective**: Requires only standard webcam vs expensive IMU/haptic systems
- **Real-time Performance**: 30 FPS hand tracking with optimized control loops
- **Independent Arm Management**: Pause individual arms while operating the other

## System Requirements

### Software Dependencies
```
Python 3.8+
OpenCV 4.5+
MediaPipe 0.10+
pyniryo
numpy
```

### Hardware Requirements
- 2x Niryo Ned2 robotic arms
- Standard webcam (640x480 minimum resolution)
- Computer with at least 8GB RAM
- Network connection to robots

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kvngAiseal/Bimanual-Teleoperation-of-Ned-2-robots-using-Mediapipe.git
cd Bimanual-Teleoperation-of-Ned-2-robots-using-Mediapipe
```

2. **Install dependencies:**
```bash
pip install opencv-python
pip install mediapipe
pip install pyniryo
pip install numpy
```

Or install all at once:
```bash
pip install opencv-python mediapipe pyniryo numpy
```

3. **Configure robot IP addresses:**
Edit `config.py` and update the robot IP addresses:
```python
ROBOT_IP_ADDRESS_LEFT = "192.168.8.147"
ROBOT_IP_ADDRESS_RIGHT = "192.168.8.146"
```

## Hardware Setup

1. **Robot Positioning**: Place the two Ned2 robots at safe distances to prevent collision
2. **Camera Setup**: Position webcam to capture both hands in the frame
3. **Network Connection**: Ensure both robots are connected to the same network as your computer
4. **Workspace Configuration**: Clear the robot workspace of obstacles

## Usage

### Basic Operation

1. **Start the system:**
```bash
python main.py
```

2. **Follow the tutorial sequence:**
   - Place hands in designated boxes for calibration
   - Complete gesture recognition tests for both arms
   - System automatically transitions to teleoperation mode

3. **Control the robots:**
   - Move hands naturally to control robot arms
   - Use gestures to switch modes and control grippers
   - Use peace sign to pause/resume operation

### Quick Start Guide

1. Run `python main.py`
2. Position hands in the blue/yellow boxes when prompted
3. Hold steady for 5 seconds for calibration
4. Complete the tutorial for both arms
5. Begin teleoperation with natural hand movements

## System Architecture

The system consists of four main modules:

### 1. Hand Tracking (`hand_tracking.py`)
- MediaPipe integration for 21-landmark hand detection
- Real-time gesture classification
- Motion smoothing and filtering

### 2. Robot Controller (`robot_controller.py`)
- Individual robot connection management
- Threaded control loops for responsive operation
- Mode switching between arm and gripper control

### 3. Configuration (`config.py`)
- System parameters and workspace boundaries
- Robot poses and communication settings
- State machine definitions

### 4. Main Controller (`main.py`)
- State machine orchestration
- User interface and tutorial system
- Coordinate transformation and command generation

## Gesture Commands

| Gesture | Function | Duration |
|---------|----------|----------|
| **Open Hand** | Open gripper | Instant |
| **Closed Hand** | Close gripper / Pause arm | Instant |
| **Pointing** | Switch between arm/gripper modes | Instant |
| **Peace Sign** | Global pause/resume | 3 seconds |
| **Thumbs Up** | Home position | 3 seconds |

## Configuration

### Workspace Boundaries
```python
# Robot workspace limits
ROBOT_MIN_X, ROBOT_MAX_X = 0.10, 0.35
ROBOT_MIN_Y, ROBOT_MAX_Y = -0.20, 0.20
ROBOT_MIN_Z, ROBOT_MAX_Z = 0.10, 0.30

# Hand tracking area
LEFT_HAND_MIN_X, LEFT_HAND_MAX_X = 0.05, 0.45
RIGHT_HAND_MIN_X, RIGHT_HAND_MAX_X = 0.55, 0.95
```

### Control Parameters
```python
# Motion smoothing and control
COMMAND_INTERVAL = 0.5  # seconds
SMOOTHING_FACTOR = 1.0
GESTURE_HOLD_DURATION = 3.0  # seconds
```

## Experimental Results

Based on user studies with 5 participants:

### Single-Arm Performance
- **Success Rate**: 68.75% (increased to 80% by trial 3)
- **Average Completion Time**: 587.5 seconds
- **Learning Effect**: Significant improvement from trial 1 to trial 3

### Bimanual Performance
- **Success Rate**: 13.3% full completion, 40% partial success
- **Independent Pause Usage**: 100% of participants actively used this feature
- **Key Finding**: Steeper learning curve but participants found the system enjoyable

### System Performance
- **Gesture Recognition Accuracy**: 92.3% (excluding thumbs-up limitations)
- **Hand Tracking Frequency**: Stable 30 FPS
- **Control Latency**: <50ms

## Troubleshooting

### Common Issues

**Hand tracking not working:**
- Ensure adequate lighting
- Check camera permissions
- Verify MediaPipe installation

**Robot connection failed:**
- Check IP addresses in config.py
- Verify network connectivity
- Restart robot controllers

**Gesture recognition issues:**
- Improve lighting conditions
- Ensure hands are within tracking boundaries
- Check for camera focus issues

**High latency:**
- Close unnecessary applications
- Check network bandwidth
- Verify camera resolution settings

### Known Limitations

- Thumbs-up gesture has angle-dependent detection issues
- Performance degrades under very bright lighting
- Requires clear camera view of both hands

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Setup

1. Install core dependencies:
```bash
pip install opencv-python mediapipe pyniryo numpy
```

2. For development, you may also want:
```bash
pip install pytest  # for testing
pip install black   # for code formatting
pip install flake8  # for linting
```

2. Run tests (if you add test files):
```bash
python -m pytest tests/
```

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bimanual_teleoperation_2025,
  title={Vision-Based Bimanual Teleoperation of Robotic Arms Using MediaPipe: A Novel Approach to Addressing Control Limitations},
  author={[Adebolanle Okuboyejo]},
  year={2025},
  url={https://github.com/kvngAiseal/Bimanual-Teleoperation-of-Ned-2-robots-using-Mediapipe}
}
```

## Acknowledgments

- Google MediaPipe team for the hand tracking framework
- Niryo for the robotic arm platform
- Research participants who provided valuable feedback

## Contact

For questions or support, please open an issue or contact [oluwaseyiokus@gmail.com].

---

**⚠️ Safety Notice**: Always ensure proper safety measures when operating robotic arms. Maintain clear workspace boundaries and keep emergency stop accessible.