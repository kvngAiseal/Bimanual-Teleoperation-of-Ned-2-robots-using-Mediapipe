import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import numpy as np
import time
import math

# ===== CONFIGURATION =====
ROBOT_IP_ADDRESS = "192.168.8.146"
CONTROL_INTERVAL = 0.05         # seconds between updates
VELOCITY_SCALE = 0.002          # pixel → meters
MAX_DELTA = 0.01                # max movement per axis per update (meters)
DEAD_ZONE_RADIUS = 30           # pixels
FIXED_ORIENTATION = [0.0, 1.57, 0.0]  # roll, pitch, yaw
# ==========================

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Connect to robot
print("Connecting to robot...")
robot = NiryoRobot(ROBOT_IP_ADDRESS)
robot.calibrate_auto()
print("Robot ready.")

# Webcam
cap = cv2.VideoCapture(0)

# Tracking state
prev_wrist_px = None
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Wrist landmark
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_px = np.array([w * wrist.x, h * wrist.y])

            # Time delta
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            # Velocity calculation
            if prev_wrist_px is not None and dt > 0:
                delta_px = wrist_px - prev_wrist_px
                speed_px = delta_px / dt

                # Dead zone
                if np.linalg.norm(speed_px) < DEAD_ZONE_RADIUS:
                    speed_px = np.array([0.0, 0.0])

                # Convert to robot-space velocity
                dx = max(min(speed_px[0] * VELOCITY_SCALE, MAX_DELTA), -MAX_DELTA)
                dy = max(min(-speed_px[1] * VELOCITY_SCALE, MAX_DELTA), -MAX_DELTA)  # invert Y

                # Get current pose
                pose = robot.get_pose()
                current_position = np.array(pose[:3])  # x, y, z

                # Update position
                new_position = current_position + np.array([dx, dy, 0.0])

                # Clamp workspace (optional)
                new_position = np.clip(new_position, [0.1, -0.2, 0.05], [0.4, 0.2, 0.3])

                # Build target pose
                target_pose = new_position.tolist() + FIXED_ORIENTATION

                # Send move command
                try:
                    robot.move_pose(target_pose)
                except Exception as e:
                    print("Move failed:", e)

            prev_wrist_px = wrist_px

        cv2.circle(frame, (w // 2, h // 2), DEAD_ZONE_RADIUS, (0, 255, 0), 2)
        cv2.imshow("Wrist Velocity Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        time.sleep(CONTROL_INTERVAL)

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        robot.stop_move()
    except:
        pass
    robot.close_connection()
    hands.close()
    print("Disconnected.")
