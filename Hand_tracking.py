import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time
import math

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"

# --- Robot Command Rate ---
COMMANDS_PER_SECOND = 10
COMMAND_INTERVAL = 1.0 / COMMANDS_PER_SECOND

# --- Smoothing Factor ---
SMOOTHING_FACTOR = 0.2

# --- Workspace Settings ---
ROBOT_MIN_X = 0.20 # Forward/Backward
ROBOT_MAX_X = 0.35
ROBOT_MIN_Y = -0.15 # Left/Right
ROBOT_MAX_Y = 0.15
ROBOT_MIN_Z = 0.15 # Down/Up
ROBOT_MAX_Z = 0.30 

# --- Hand Tracking Settings ---
HAND_TRACKING_MIN_X = 0.2 # Left/Right on screen
HAND_TRACKING_MAX_X = 0.8
HAND_TRACKING_MIN_Y = 0.2 # Up/Down on screen
HAND_TRACKING_MAX_Y = 0.8
HAND_MIN_SIZE = 0.10 # Hand appears small (far)
HAND_MAX_SIZE = 0.25 # Hand appears large (close)
# ===================================

# MediaPipe setup and helper functions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

def scale_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_hand_size(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return math.sqrt((wrist.x - mcp_middle.x)**2 + (wrist.y - mcp_middle.y)**2)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    robot = None
    smoothed_x, smoothed_y, smoothed_z = None, None, None
    last_command_time = 0

    try:
        print("--- Connecting to robot ---")
        robot = NiryoRobot(ROBOT_IP_ADDRESS)
        print("Connected successfully.")
        
        start_pose = [0.25, 0.0, 0.2, 0.0, 1.57, 0.0]
        robot.move_pose(start_pose)
        print("Robot at start pose. Show your hand to begin tracking.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_size = get_hand_size(hand_landmarks)

                    raw_x = scale_value(hand_size, HAND_MIN_SIZE, HAND_MAX_SIZE, ROBOT_MAX_X, ROBOT_MIN_X) 
                    raw_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                    raw_z = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_Z, ROBOT_MIN_Z)
                    
                    if smoothed_x is None:
                        smoothed_x, smoothed_y, smoothed_z = raw_x, raw_y, raw_z
                    else:
                        smoothed_x = (SMOOTHING_FACTOR * raw_x) + ((1 - SMOOTHING_FACTOR) * smoothed_x)
                        smoothed_y = (SMOOTHING_FACTOR * raw_y) + ((1 - SMOOTHING_FACTOR) * smoothed_y)
                        smoothed_z = (SMOOTHING_FACTOR * raw_z) + ((1 - SMOOTHING_FACTOR) * smoothed_z)
                    
                    current_time = time.time()
                    if (current_time - last_command_time) > COMMAND_INTERVAL:
                        target_pose = [smoothed_x, smoothed_y, smoothed_z, 0.0, 1.57, 0.0]
                        robot.move_pose(target_pose)
                        last_command_time = current_time
            
            cv2.imshow("Hand Tracking Control", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    except Exception as e:
        print("An error occurred:", e)

    finally:
        print("--- Cleaning up ---")
        if robot:
            robot.go_to_sleep()
            robot.close_connection()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released.")

if __name__ == "__main__":
    main()