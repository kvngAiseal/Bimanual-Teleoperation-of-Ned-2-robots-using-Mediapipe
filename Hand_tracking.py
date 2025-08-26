import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"

# --- Workspace Settings ---
ROBOT_MIN_X = 0.20 # Forward/Backward
ROBOT_MAX_X = 0.35
ROBOT_MIN_Y = -0.15 # Left/Right
ROBOT_MAX_Y = 0.15
ROBOT_Z_POSITION = 0.25 

# --- Hand Tracking Settings ---
HAND_TRACKING_MIN_X = 0.2 # Left/Right on screen
HAND_TRACKING_MAX_X = 0.8
HAND_TRACKING_MIN_Y = 0.2 # Up/Down on screen
HAND_TRACKING_MAX_Y = 0.8
# ===================================

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

def scale_value(value, in_min, in_max, out_min, out_max):
    """Helper function to map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main():
    cap = cv2.VideoCapture(0)
    robot = None

    try:
        print("--- Connecting to robot ---")
        robot = NiryoRobot(ROBOT_IP_ADDRESS)
        print("Connected successfully.")
        
        start_pose = [0.25, 0.0, 0.2, 0.0, 1.57, 0.0]
        robot.move_pose(start_pose)
        print("Robot at start pose. Show your hand to begin tracking.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    
                    # Hand's Up/Down (wrist.y) controls Robot's Forward/Backward (target_x)
                    target_x = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_X, ROBOT_MIN_X) # Flipped
                    
                    # Hand's Left/Right (wrist.x) controls Robot's Left/Right (target_y)
                    target_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                    
                    target_pose = [target_x, target_y, ROBOT_Z_POSITION, 0.0, 1.57, 0.0]
                    robot.move_pose(target_pose)
            
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