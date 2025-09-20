# config.py

import math

# ========== ROBOT & COMMUNICATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"
COMMAND_INTERVAL = 0.5  # Seconds between commands in the control thread
SMOOTHING_FACTOR = 1.0   # How much to smooth robot movement (1.0 = no smoothing)

# --- Workspace & Hand Tracking Values ---
# Robot's physical workspace in meters
ROBOT_MIN_X, ROBOT_MAX_X = 0.10, 0.35
ROBOT_MIN_Y, ROBOT_MAX_Y = -0.15, 0.15
ROBOT_MIN_Z, ROBOT_MAX_Z = 0.15, 0.30 

# Camera's active tracking area (normalized screen coordinates)
HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X = 0.2, 0.8
HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y = 0.2, 0.8

# Hand size mapping for depth (X-axis)
HAND_MIN_SIZE, HAND_MAX_SIZE = 0.15, 0.25

# --- Poses ---
START_POSE = [0.30, 0.0, 0.3, 0.0, 1.57, 0.0]
HOME_POSE = [0.14, 0.00, 0.20, 0.00, 0.75, 0.00]

# --- UI & Interaction ---
# Tutorial start box
BOX_TOP_LEFT = (0.4, 0.3)
BOX_BOTTOM_RIGHT = (0.6, 0.7)
HOLD_DURATION = 5.0 # seconds

# Gesture hold time for pause, resume, and home
GESTURE_HOLD_DURATION = 3.0 # seconds

# Menu selection boxes
MENU_HOLD_DURATION = 3.0 # seconds
RESUME_BOX_TL = (0.1, 0.3)
RESUME_BOX_BR = (0.4, 0.7)
EXIT_BOX_TL = (0.6, 0.3)
EXIT_BOX_BR = (0.9, 0.7)

# --- State Machine Definitions ---
STATE_WAITING_FOR_HAND = 0
STATE_TEST_MODE_SWITCH = 1
STATE_TEST_GRIPPER_OPEN = 2
STATE_TEST_GRIPPER_CLOSE = 3
STATE_TEST_GRIPPER_ROTATION = 4
STATE_TEST_HOME = 5
STATE_TEST_PAUSE = 6
STATE_TEST_RESUME = 7
STATE_TELEOP_ACTIVE = 8
STATE_CHOICE_MENU = 9
STATE_EXITING = 10