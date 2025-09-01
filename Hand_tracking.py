import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time
import math

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"

# --- Robot Command Rate (reasonable rate to prevent overwhelming) ---
COMMAND_INTERVAL = 1.5  # 10 FPS - slower but more stable

# --- Light Smoothing (much less aggressive than before) ---
SMOOTHING_FACTOR = 1.0  # Very light smoothing to reduce jitter

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

# --- Camera optimization (minimal changes) ---
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for camera smoothness
# ===================================

# MediaPipe setup - keep it simple like the original
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.7,  # Same as original
    min_tracking_confidence=0.7    # Same as original
)

def scale_value(value, in_min, in_max, out_min, out_max):
    """Helper function to map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_hand_size(hand_landmarks):
    """Calculate hand size for depth estimation"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return math.sqrt((wrist.x - mcp_middle.x)**2 + (wrist.y - mcp_middle.y)**2)

def main():
    # Camera setup with minimal optimization
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Only this optimization to reduce camera lag
    
    robot = None
    
    # Simple smoothing variables and current position tracking
    smoothed_x, smoothed_y, smoothed_z = None, None, None
    current_robot_pos = [0.25, 0.0, 0.2]  # Track current robot position
    last_command_time = 0
    frame_count = 0

    try:
        print("--- Connecting to robot ---")
        robot = NiryoRobot(ROBOT_IP_ADDRESS)
        print("Connected successfully.")
        
        start_pose = [0.25, 0.0, 0.2, 0.0, 1.57, 0.0]
        robot.move_pose(start_pose)
        current_robot_pos = start_pose[:3]  # Initialize tracking position
        print("Robot at start pose. Show your hand to begin tracking.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Clear buffer to get latest frame (reduces camera lag)
            if cap.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
                cap.grab()

            frame = cv2.flip(frame, 1)
            
            # Process every Nth frame to smooth camera feed
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks every frame (no flickering)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        hand_size = get_hand_size(hand_landmarks)

                        # Calculate target positions (same as your original extended version)
                        raw_x = scale_value(hand_size, HAND_MIN_SIZE, HAND_MAX_SIZE, ROBOT_MAX_X, ROBOT_MIN_X) 
                        raw_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                        raw_z = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_Z, ROBOT_MIN_Z)
                        
                        # Apply very light smoothing (much lighter than complex versions)
                        if smoothed_x is None:
                            smoothed_x, smoothed_y, smoothed_z = raw_x, raw_y, raw_z
                        else:
                            smoothed_x = (SMOOTHING_FACTOR * raw_x) + ((1 - SMOOTHING_FACTOR) * smoothed_x)
                            smoothed_y = (SMOOTHING_FACTOR * raw_y) + ((1 - SMOOTHING_FACTOR) * smoothed_y)
                            smoothed_z = (SMOOTHING_FACTOR * raw_z) + ((1 - SMOOTHING_FACTOR) * smoothed_z)
                        
                        # Rate limiting (prevent overwhelming robot)
                        current_time = time.time()
                        if (current_time - last_command_time) > COMMAND_INTERVAL:
                            # Use simple move_pose but with smaller steps for smoother movement
                            target_pose = [smoothed_x, smoothed_y, smoothed_z, 0.0, 1.57, 0.0]
                            
                            try:
                                robot.move_pose(target_pose)
                                current_robot_pos = [smoothed_x, smoothed_y, smoothed_z]
                            except Exception as e:
                                print(f"Robot movement error: {e}")
                                # Don't break the loop, just continue
                            
                            last_command_time = current_time
            
            frame_count += 1
            cv2.imshow("Hand Tracking Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                break

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