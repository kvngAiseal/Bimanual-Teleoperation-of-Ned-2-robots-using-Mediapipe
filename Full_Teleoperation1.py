import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time
import math
import threading

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"
# Using your tuned values
COMMAND_INTERVAL = 0.5
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
HOLD_DURATION = 5.0 # seconds
PAUSE_HOLD_DURATION = 3.0 # seconds for pause/resume
# ===================================

# --- Global variables for threads ---
latest_target_pose = None
latest_gesture = None
is_paused = False 
pose_lock = threading.Lock()
stop_threads = False

# --- State Machine for Tutorial ---
STATE_WAITING_FOR_HAND = 0
STATE_TEST_MODE_SWITCH = 1
STATE_TEST_GRIPPER_OPEN = 2
STATE_TEST_GRIPPER_CLOSE = 3
STATE_TEST_PAUSE = 4
STATE_TELEOP_ACTIVE = 5

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
    
    is_peace = (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                lm[mp_hands.HandLandmark.PINKY_TIP].y > lm[mp_hands.HandLandmark.PINKY_PIP].y)
    if is_peace:
        return "Peace"

    is_pointing = (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                   lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                   lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                   lm[mp_hands.HandLandmark.PINKY_TIP].y > lm[mp_hands.HandLandmark.PINKY_PIP].y)
    if is_pointing:
        return "Pointing"

    is_open = (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
               lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
               lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
               lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y)
    if is_open:
        return "Open Hand"
    
    return "Closed Hand"

def robot_control_thread(robot):
    global latest_target_pose, latest_gesture, is_paused, stop_threads
    
    smoothed_x, smoothed_y, smoothed_z = None, None, None
    last_sent_gesture = None
    
    MODE_ARM_CONTROL = "ARM"
    MODE_GRIPPER_CONTROL = "GRIPPER"
    current_mode = MODE_ARM_CONTROL
    last_toggle_time = 0

    while not stop_threads:
        if is_paused:
            time.sleep(COMMAND_INTERVAL)
            continue

        with pose_lock:
            target_pose_raw = latest_target_pose
            current_gesture = latest_gesture
        
        if current_gesture == "Pointing" and (time.time() - last_toggle_time > 1.5):
            current_mode = MODE_GRIPPER_CONTROL if current_mode == MODE_ARM_CONTROL else MODE_ARM_CONTROL
            print(f"Switched to {current_mode} CONTROL mode")
            last_toggle_time = time.time()
        
        if current_mode == MODE_ARM_CONTROL and target_pose_raw:
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

        elif current_mode == MODE_GRIPPER_CONTROL and current_gesture and current_gesture != last_sent_gesture:
            if current_gesture in ["Open Hand", "Closed Hand"]:
                try:
                    if current_gesture == "Open Hand":
                        robot.open_gripper(speed=500)
                    else:
                        robot.close_gripper(speed=500)
                    last_sent_gesture = current_gesture
                except Exception as e:
                    print(f"Gripper command error: {e}")
        
        time.sleep(COMMAND_INTERVAL)

def main():
    global latest_target_pose, latest_gesture, is_paused, stop_threads
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    robot = None
    control_thread = None
    current_state = STATE_WAITING_FOR_HAND
    hand_in_box_start_time = None
    pause_gesture_start_time = None # New timer for the pause gesture
    
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
            
            # --- Tutorial State Machine Logic ---
            if current_state < STATE_TELEOP_ACTIVE:
                # ... (The tutorial logic is unchanged) ...
                if current_state == STATE_WAITING_FOR_HAND:
                    instruction_text = "Place Hand in Box to Start"
                    start_point = (int(BOX_TOP_LEFT[0] * frame_width), int(BOX_TOP_LEFT[1] * frame_height))
                    end_point = (int(BOX_BOTTOM_RIGHT[0] * frame_width), int(BOX_BOTTOM_RIGHT[1] * frame_height))
                    cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
                    if hand_is_visible:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        if (BOX_TOP_LEFT[0] < wrist.x < BOX_BOTTOM_RIGHT[0]) and (BOX_TOP_LEFT[1] < wrist.y < BOX_BOTTOM_RIGHT[1]):
                            if hand_in_box_start_time is None: hand_in_box_start_time = time.time()
                            time_in_box = time.time() - hand_in_box_start_time
                            instruction_text = f"Hold Steady... {int(HOLD_DURATION - time_in_box)}s"
                            if time_in_box > HOLD_DURATION:
                                print("Step 1 Complete: Moving to start pose.")
                                robot.move_pose([0.30, 0.0, 0.3, 0.0, 1.57, 0.0])
                                current_state = STATE_TEST_MODE_SWITCH
                        else: hand_in_box_start_time = None
                    else: hand_in_box_start_time = None

                elif current_state == STATE_TEST_MODE_SWITCH:
                    instruction_text = "Raise Index Finger Mode Switch Test"
                    if hand_is_visible and classify_gesture(hand_landmarks) == "Pointing":
                        print("Step 2 Complete: Mode switch tested.")
                        current_state = STATE_TEST_GRIPPER_OPEN

                elif current_state == STATE_TEST_GRIPPER_OPEN:
                    instruction_text = "OPEN HAND to Open Gripper"
                    if hand_is_visible and classify_gesture(hand_landmarks) == "Open Hand":
                        robot.open_gripper(speed=500)
                        print("Step 3 Complete: Gripper open tested.")
                        current_state = STATE_TEST_GRIPPER_CLOSE
                
                elif current_state == STATE_TEST_GRIPPER_CLOSE:
                    instruction_text = "Great! Make a FIST to Close Gripper"
                    if hand_is_visible and classify_gesture(hand_landmarks) == "Closed Hand":
                        robot.close_gripper(speed=500)
                        print("Step 4 Complete: Gripper close tested.")
                        current_state = STATE_TEST_PAUSE

                elif current_state == STATE_TEST_PAUSE:
                    instruction_text = "Finally, show a PEACE SIGN to Pause"
                    if hand_is_visible and classify_gesture(hand_landmarks) == "Peace":
                        print("Step 5 Complete: Pause tested.")
                        time.sleep(1)
                        print("Tutorial Complete! Starting teleoperation.")
                        control_thread = threading.Thread(target=robot_control_thread, args=(robot,), daemon=True)
                        control_thread.start()
                        current_state = STATE_TELEOP_ACTIVE

            elif current_state == STATE_TELEOP_ACTIVE:
                # --- Main Teleoperation Logic ---
                gesture = "No Hand"
                if hand_is_visible:
                    gesture = classify_gesture(hand_landmarks)
                
                # --- "Hold to Pause/Resume" Logic ---
                if gesture == "Peace":
                    if pause_gesture_start_time is None:
                        # Start the timer
                        pause_gesture_start_time = time.time()
                    
                    time_held = time.time() - pause_gesture_start_time
                    
                    # Update instruction text with countdown
                    if is_paused:
                        instruction_text = f"Resuming in {int(PAUSE_HOLD_DURATION - time_held)}s"
                    else:
                        instruction_text = f"Pausing in {int(PAUSE_HOLD_DURATION - time_held)}s"

                    # Check if held long enough
                    if time_held > PAUSE_HOLD_DURATION:
                        is_paused = not is_paused # Toggle the pause state
                        print(f"Teleoperation {'PAUSED' if is_paused else 'RESUMED'}")
                        pause_gesture_start_time = None # Reset timer to prevent rapid toggling
                else:
                    # If gesture is not "Peace", reset the timer
                    pause_gesture_start_time = None

                # --- Update display text and send data to thread ---
                if is_paused:
                    if pause_gesture_start_time is None: # Only show this message if timer isn't active
                         instruction_text = "PAUSED (Hold Peace to Resume)"
                else:
                    if pause_gesture_start_time is None:
                        instruction_text = "Teleoperation Active"
                    
                    # Only send data to thread if not paused
                    if hand_is_visible:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        hand_size = get_hand_size(hand_landmarks)
                        raw_x = scale_value(hand_size, HAND_MIN_SIZE, HAND_MAX_SIZE, ROBOT_MAX_X, ROBOT_MIN_X) 
                        raw_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                        raw_z = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_Z, ROBOT_MIN_Z)
                        with pose_lock:
                            latest_target_pose = [raw_x, raw_y, raw_z]
                            latest_gesture = gesture
                    else:
                        with pose_lock:
                            latest_gesture = "No Hand"

            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking Control", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    except Exception as e:
        print("An error occurred:", e)

    finally:
        print("--- Cleaning up ---")
        stop_threads = True
        if control_thread is not None:
            control_thread.join(timeout=2.0)
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