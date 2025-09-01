import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time
import math
import threading

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"

# --- Your Tuned Values ---
COMMAND_INTERVAL = 1.5
SMOOTHING_FACTOR = 1.0

# --- Workspace & Hand Tracking Values ---
ROBOT_MIN_X, ROBOT_MAX_X = 0.20, 0.35
ROBOT_MIN_Y, ROBOT_MAX_Y = -0.15, 0.15
ROBOT_MIN_Z, ROBOT_MAX_Z = 0.15, 0.30 

HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X = 0.2, 0.8
HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y = 0.2, 0.8
HAND_MIN_SIZE, HAND_MAX_SIZE = 0.10, 0.25

# --- Readiness Test Config ---
BOX_TOP_LEFT = (0.4, 0.3)
BOX_BOTTOM_RIGHT = (0.6, 0.7)
HOLD_DURATION = 3.0 # seconds
# ===================================

# --- Global variables for threads ---
latest_target_pose = None
latest_gesture = None
pose_lock = threading.Lock()
stop_threads = False

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

def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers_extended = (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                        lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y)
    return "Open Hand" if fingers_extended else "Closed Hand"

def robot_control_thread(robot):
    global latest_target_pose, latest_gesture, stop_threads
    smoothed_x, smoothed_y, smoothed_z = None, None, None
    last_sent_gesture = None
    
    while not stop_threads:
        with pose_lock:
            target_pose_raw = latest_target_pose
            current_gesture = latest_gesture
        
        # --- Arm Movement Control ---
        if target_pose_raw:
            raw_x, raw_y, raw_z = target_pose_raw
            if smoothed_x is None:
                smoothed_x, smoothed_y, smoothed_z = raw_x, raw_y, raw_z
            else:
                smoothed_x = (SMOOTHING_FACTOR * raw_x) + ((1 - SMOOTHING_FACTOR) * smoothed_x)
                smoothed_y = (SMOOTHING_FACTOR * raw_y) + ((1 - SMOOTHING_FACTOR) * smoothed_y)
                smoothed_z = (SMOOTHING_FACTOR * raw_z) + ((1 - SMOOTHING_FACTOR) * smoothed_z)
            try:
                robot.move_pose([smoothed_x, smoothed_y, smoothed_z, 0.0, 1.57, 0.0])
            except Exception as e:
                print(f"Robot movement error: {e}")

        # --- Gripper Control ---
        if current_gesture and current_gesture != last_sent_gesture:
            try:
                if current_gesture == "Open Hand":
                    robot.open_gripper(speed=500)
                else: # Closed Hand
                    robot.close_gripper(speed=500)
                last_sent_gesture = current_gesture
            except Exception as e:
                print(f"Gripper command error: {e}")

        time.sleep(COMMAND_INTERVAL)

def main():
    global latest_target_pose, latest_gesture, stop_threads
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    robot = None
    teleop_started = False
    hand_in_box_start_time = None
    
    try:
        print("--- Connecting to robot ---")
        robot = NiryoRobot(ROBOT_IP_ADDRESS)
        robot.calibrate_auto()
        robot.update_tool()
        print("Connected successfully.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_height, frame_width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            instruction_text = ""
            hand_is_visible = results.multi_hand_landmarks is not None

            if hand_is_visible:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not teleop_started:
                # --- Readiness Test Logic ---
                instruction_text = "Place Hand in Box to Start"
                start_point = (int(BOX_TOP_LEFT[0] * frame_width), int(BOX_TOP_LEFT[1] * frame_height))
                end_point = (int(BOX_BOTTOM_RIGHT[0] * frame_width), int(BOX_BOTTOM_RIGHT[1] * frame_height))
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
                
                if hand_is_visible:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    if (BOX_TOP_LEFT[0] < wrist.x < BOX_BOTTOM_RIGHT[0]) and (BOX_TOP_LEFT[1] < wrist.y < BOX_BOTTOM_RIGHT[1]):
                        if hand_in_box_start_time is None:
                            hand_in_box_start_time = time.time()
                        time_in_box = time.time() - hand_in_box_start_time
                        instruction_text = f"Hold Steady... {int(HOLD_DURATION - time_in_box)}s"
                        if time_in_box > HOLD_DURATION:
                            print("Readiness test passed. Starting teleoperation.")
                            start_pose = [0.30, 0.0, 0.3, 0.0, 1.57, 0.0]
                            robot.move_pose(start_pose)
                            control_thread = threading.Thread(target=robot_control_thread, args=(robot,), daemon=True)
                            control_thread.start()
                            teleop_started = True
                    else:
                        hand_in_box_start_time = None
                else:
                    hand_in_box_start_time = None
            else:
                # --- Teleoperation is Active ---
                instruction_text = "Teleoperation Active"
                if hand_is_visible:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_size = get_hand_size(hand_landmarks)
                    gesture = classify_gesture(hand_landmarks)
                    
                    raw_x = scale_value(hand_size, HAND_MIN_SIZE, HAND_MAX_SIZE, ROBOT_MAX_X, ROBOT_MIN_X) 
                    raw_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                    raw_z = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_Z, ROBOT_MIN_Z)
                    
                    # Update shared variables for the control thread
                    with pose_lock:
                        latest_target_pose = [raw_x, raw_y, raw_z]
                        latest_gesture = gesture
            
            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking Control", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    except Exception as e:
        print("An error occurred:", e)

    finally:
        print("--- Cleaning up ---")
        stop_threads = True
        time.sleep(0.5)
        if robot:
            robot.move_pose([0.30, 0.0, 0.3, 0.0, 1.57, 0.0])
            robot.go_to_sleep()
            robot.close_connection()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released.")

if __name__ == "__main__":
    main()