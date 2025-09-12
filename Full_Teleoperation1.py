import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time
import math
import threading

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"
COMMAND_INTERVAL = 0.5
SMOOTHING_FACTOR = 1.0

# --- Workspace & Hand Tracking Values ---
ROBOT_MIN_X, ROBOT_MAX_X = 0.10, 0.35
ROBOT_MIN_Y, ROBOT_MAX_Y = -0.15, 0.15
ROBOT_MIN_Z, ROBOT_MAX_Z = 0.15, 0.30 

HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X = 0.2, 0.8
HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y = 0.2, 0.8
HAND_MIN_SIZE, HAND_MAX_SIZE = 0.15, 0.25

# --- Readiness Test & Gesture Config ---
BOX_TOP_LEFT = (0.4, 0.3)
BOX_BOTTOM_RIGHT = (0.6, 0.7)
HOLD_DURATION = 5.0 # seconds
GESTURE_HOLD_DURATION = 3.0 # seconds for pause, resume, and home

# --- Menu Box Configuration ---
MENU_HOLD_DURATION = 3.0 # seconds to hold hand in a menu box
RESUME_BOX_TL = (0.1, 0.3) # Top-Left for Resume box (normalized)
RESUME_BOX_BR = (0.4, 0.7) # Bottom-Right for Resume box (normalized)
EXIT_BOX_TL = (0.6, 0.3)   # Top-Left for Exit box (normalized)
EXIT_BOX_BR = (0.9, 0.7)   # Bottom-Right for Exit box (normalized)

START_POSE = [0.30, 0.0, 0.3, 0.0, 1.57, 0.0]
HOME_POSE = [0.14, 0.00, 0.20, 0.00, 0.75, 0.00]
# ===================================

# --- Global variables for threads ---
latest_target_pose = None
latest_gesture = None
is_paused = False 
pose_lock = threading.Lock()
stop_threads = False # Flag to stop all threads

# --- State Machine ---
STATE_WAITING_FOR_HAND = 0
STATE_TEST_MODE_SWITCH = 1
STATE_TEST_GRIPPER_OPEN = 2
STATE_TEST_GRIPPER_CLOSE = 3
STATE_TEST_HOME = 4
STATE_TEST_PAUSE = 5
STATE_TEST_RESUME = 6
STATE_TELEOP_ACTIVE = 7
STATE_CHOICE_MENU = 8 # New state for menu selection
STATE_EXITING = 9 # State to gracefully exit

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

def classify_gesture(landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers_extended = []

    # Thumb
    thumb_tip = landmarks.landmark[tip_ids[0]]
    thumb_ip  = landmarks.landmark[tip_ids[0] - 1]
    if hand_label == "Right":
        thumb_ext = thumb_tip.x < thumb_ip.x
    else:
        thumb_ext = thumb_tip.x > thumb_ip.x
    fingers_extended.append(thumb_ext)

    # Other Fingers
    for tip_id in tip_ids[1:]:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[tip_id - 2]
        fingers_extended.append(tip.y < pip.y)

    # Gesture Patterns
    if fingers_extended[0] and not any(fingers_extended[1:]):
        return "Thumbs Up"
    if fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[i] for i in [0, 3, 4]):
        return "Peace"
    if fingers_extended[1] and not any(fingers_extended[i] for i in [0, 2, 3, 4]):
        return "Pointing"
    if fingers_extended[4] and not any(fingers_extended[i] for i in [0, 1, 2, 3]):
        return "Pinky"  # NEW: Pinky finger gesture
    if all(fingers_extended):
        return "Open Hand"
    if not any(fingers_extended):
        return "Closed Hand"
    return "Unclear"

def robot_control_thread(robot):
    global latest_target_pose, latest_gesture, is_paused, stop_threads
    
    smoothed_x, smoothed_y, smoothed_z = None, None, None
    last_sent_gesture = None
    
    MODE_ARM_CONTROL = "ARM"
    MODE_GRIPPER_CONTROL = "GRIPPER"
    MODE_GRIPPER_ROTATION = "ROTATION"  # NEW: Rotation mode
    current_mode = MODE_ARM_CONTROL
    last_toggle_time = 0
    
    # Store current pose for rotation mode
    current_robot_pose = list(START_POSE)  # Initialize with start pose

    while not stop_threads:
        if is_paused:
            time.sleep(COMMAND_INTERVAL)
            continue

        with pose_lock:
            target_pose_raw = latest_target_pose
            current_gesture = latest_gesture
        
        # Mode switching logic
        if current_gesture == "Pointing" and (time.time() - last_toggle_time > 1.5):
            if current_mode == MODE_ARM_CONTROL:
                current_mode = MODE_GRIPPER_CONTROL
            elif current_mode == MODE_GRIPPER_CONTROL:
                current_mode = MODE_ARM_CONTROL
            elif current_mode == MODE_GRIPPER_ROTATION:
                current_mode = MODE_ARM_CONTROL
            print(f"Switched to {current_mode} mode")
            last_toggle_time = time.time()
        
        # NEW: Pinky switches to rotation mode from any mode
        if current_gesture == "Pinky" and (time.time() - last_toggle_time > 1.5):
            current_mode = MODE_GRIPPER_ROTATION
            print(f"Switched to {current_mode} mode")
            last_toggle_time = time.time()
        
        # ARM CONTROL MODE
        if current_mode == MODE_ARM_CONTROL and target_pose_raw:
            raw_x, raw_y, raw_z = target_pose_raw
            if smoothed_x is None:
                smoothed_x, smoothed_y, smoothed_z = raw_x, raw_y, raw_z
            else:
                smoothed_x = (SMOOTHING_FACTOR * raw_x) + ((1 - SMOOTHING_FACTOR) * smoothed_x)
                smoothed_y = (SMOOTHING_FACTOR * raw_y) + ((1 - SMOOTHING_FACTOR) * smoothed_y)
                smoothed_z = (SMOOTHING_FACTOR * raw_z) + ((1 - SMOOTHING_FACTOR) * smoothed_z)
            
            # Update stored pose and send command
            current_robot_pose = [smoothed_x, smoothed_y, smoothed_z, 0.0, 1.57, 0.0]
            try:
                robot.move_pose(current_robot_pose)
            except Exception as e:
                print(f"Robot movement error: {e}")

        # GRIPPER CONTROL MODE
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
        
        # NEW: GRIPPER ROTATION MODE
        elif current_mode == MODE_GRIPPER_ROTATION and target_pose_raw:
            # Use hand X position to control gripper rotation
            _, hand_x_norm, _ = target_pose_raw  # This comes from wrist.x
            
            # Convert hand X position to rotation angle (-π to π)
            rotation_angle = scale_value(hand_x_norm, 
                                       ROBOT_MIN_Y, ROBOT_MAX_Y,  # Using Y range for X input
                                       -math.pi, math.pi)
            
            # Keep current X, Y, Z but update yaw rotation
            rotation_pose = current_robot_pose.copy()
            rotation_pose[5] = rotation_angle  # Update yaw (index 5)
            
            try:
                robot.move_pose(rotation_pose)
            except Exception as e:
                print(f"Robot rotation error: {e}")
        
        time.sleep(COMMAND_INTERVAL)

def main():
    global latest_target_pose, latest_gesture, is_paused, stop_threads
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    robot = None
    control_thread = None
    current_state = STATE_WAITING_FOR_HAND
    gesture_start_time = None
    menu_selection_time = None # New timer for menu selections
    
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
            hand_is_visible = results.multi_hand_landmarks and results.multi_handedness
            gesture = "No Hand" # Default to No Hand

            if hand_is_visible:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks, hand_label)
                wrist_normalized = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] # For menu box detection
            
            # --- State Machine Logic ---
            if current_state == STATE_WAITING_FOR_HAND:
                instruction_text = "Place Hand in Box to Start"
                start_point = (int(BOX_TOP_LEFT[0] * frame_width), int(BOX_TOP_LEFT[1] * frame_height))
                end_point = (int(BOX_BOTTOM_RIGHT[0] * frame_width), int(BOX_BOTTOM_RIGHT[1] * frame_height))
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
                if hand_is_visible:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    if (BOX_TOP_LEFT[0] < wrist.x < BOX_BOTTOM_RIGHT[0]) and (BOX_TOP_LEFT[1] < wrist.y < BOX_BOTTOM_RIGHT[1]):
                        if gesture_start_time is None: gesture_start_time = time.time()
                        time_held = time.time() - gesture_start_time
                        instruction_text = f"Hold Steady... {int(HOLD_DURATION - time_held)}s"
                        if time_held > HOLD_DURATION:
                            print("Step 1 Complete: Moving to start pose.")
                            robot.move_pose(START_POSE)
                            current_state = STATE_TEST_MODE_SWITCH
                            gesture_start_time = None
                    else: gesture_start_time = None
                else: gesture_start_time = None

            elif current_state == STATE_TEST_MODE_SWITCH:
                instruction_text = "Raise Index Finger to test Mode Switch"
                if gesture == "Pointing":
                    print("Step 2 Complete: Mode switch tested.")
                    current_state = STATE_TEST_GRIPPER_OPEN
            
            elif current_state == STATE_TEST_GRIPPER_OPEN:
                instruction_text = "OPEN HAND to Open Gripper"
                if gesture == "Open Hand":
                    robot.open_gripper(speed=500)
                    print("Step 3 Complete: Gripper open tested.")
                    current_state = STATE_TEST_GRIPPER_CLOSE
            
            elif current_state == STATE_TEST_GRIPPER_CLOSE:
                instruction_text = "Great! Make a FIST to Close Gripper"
                if gesture == "Closed Hand":
                    robot.close_gripper(speed=500)
                    print("Step 4 Complete: Gripper close tested.")
                    current_state = STATE_TEST_HOME
            
            elif current_state == STATE_TEST_HOME:
                instruction_text = "Hold THUMBS UP to test Go Home"
                if gesture == "Thumbs Up":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Going Home in {int(GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > GESTURE_HOLD_DURATION:
                        robot.move_pose(HOME_POSE)
                        time.sleep(1) # Wait for move to finish
                        robot.move_pose(START_POSE) # Return to start for next test
                        print("Step 5 Complete: Go Home tested.")
                        current_state = STATE_TEST_PAUSE
                        gesture_start_time = None
                else:
                    gesture_start_time = None
                    
            elif current_state == STATE_TEST_PAUSE:
                instruction_text = "Hold PEACE SIGN to test Pause"
                if gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Pausing in {int(GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > GESTURE_HOLD_DURATION:
                        print("Step 6 Complete: Pause tested.")
                        current_state = STATE_TEST_RESUME
                        gesture_start_time = None
                else:
                    gesture_start_time = None
            
            elif current_state == STATE_TEST_RESUME:
                instruction_text = "Hold PEACE SIGN again to test Resume"
                if gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Resuming in {int(GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > GESTURE_HOLD_DURATION:
                        print("Step 7 Complete: Resume tested.")
                        print("Tutorial Complete! Starting teleoperation.")
                        control_thread = threading.Thread(target=robot_control_thread, args=(robot,), daemon=True)
                        control_thread.start()
                        current_state = STATE_TELEOP_ACTIVE
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == STATE_TELEOP_ACTIVE:
                is_holding_peace = (gesture == "Peace")
                is_holding_thumbs_up = (gesture == "Thumbs Up")
                
                # Handle Peace Sign (Pause/Resume)
                if is_holding_peace:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    
                    instruction_text = f"Resuming in {int(GESTURE_HOLD_DURATION - time_held)}s" if is_paused else f"Pausing in {int(GESTURE_HOLD_DURATION - time_held)}s"
                        
                    if time_held > GESTURE_HOLD_DURATION:
                        is_paused = not is_paused
                        print(f"Teleoperation {'PAUSED' if is_paused else 'RESUMED'}")
                        gesture_start_time = None 
                else:
                    if gesture_start_time is not None and gesture != "Thumbs Up": # Reset if not holding peace or thumbs up
                        gesture_start_time = None
                
                # Handle Thumbs Up (Go Home and Menu)
                if is_holding_thumbs_up and not is_paused:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Parking Robot in {int(GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > GESTURE_HOLD_DURATION:
                        print("Thumbs Up held. Stopping teleoperation and moving to home pose.")
                        stop_threads = True # Signal the control thread to stop
                        if control_thread is not None:
                            control_thread.join(timeout=2.0) # Wait for it to finish
                        robot.move_pose(HOME_POSE) # Send final move command
                        current_state = STATE_CHOICE_MENU # Transition to new menu state
                        gesture_start_time = None
                
                # Default instruction text if no gesture is being held
                if gesture_start_time is None:
                    if is_paused:
                        instruction_text = "PAUSED (Hold Peace to Resume)"
                    else:
                        instruction_text = "Teleoperation Active (Point=Mode, Pinky=Rotate)"
                
                # Update data for the control thread only if not paused and not transitioning
                if not is_paused and hand_is_visible and current_state == STATE_TELEOP_ACTIVE:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_size = get_hand_size(hand_landmarks)
                    raw_x = scale_value(hand_size, HAND_MAX_SIZE, HAND_MIN_SIZE, ROBOT_MAX_X, ROBOT_MIN_X) 
                    raw_y = scale_value(wrist.x, HAND_TRACKING_MIN_X, HAND_TRACKING_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y)
                    raw_z = scale_value(wrist.y, HAND_TRACKING_MIN_Y, HAND_TRACKING_MAX_Y, ROBOT_MAX_Z, ROBOT_MIN_Z)
                    with pose_lock:
                        latest_target_pose = [raw_x, raw_y, raw_z]
                        latest_gesture = gesture
                else:
                    with pose_lock:
                        latest_gesture = "No Hand"

            elif current_state == STATE_CHOICE_MENU:
                instruction_text = "Choose an Option"
                
                # Draw "Resume Teleoperation" box
                resume_box_start = (int(RESUME_BOX_TL[0] * frame_width), int(RESUME_BOX_TL[1] * frame_height))
                resume_box_end = (int(RESUME_BOX_BR[0] * frame_width), int(RESUME_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, resume_box_start, resume_box_end, (0, 255, 255), 2) # Yellow box
                cv2.putText(frame, "Resume Teleoperation", (resume_box_start[0] + 10, resume_box_start[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw "Exit" box
                exit_box_start = (int(EXIT_BOX_TL[0] * frame_width), int(EXIT_BOX_TL[1] * frame_height))
                exit_box_end = (int(EXIT_BOX_BR[0] * frame_width), int(EXIT_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, exit_box_start, exit_box_end, (0, 0, 255), 2) # Red box
                cv2.putText(frame, "Exit", (exit_box_start[0] + 10, exit_box_start[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Check hand position for selection
                if hand_is_visible:
                    hand_x_norm = wrist_normalized.x
                    hand_y_norm = wrist_normalized.y

                    # Check if hand is in Resume box
                    if (RESUME_BOX_TL[0] < hand_x_norm < RESUME_BOX_BR[0]) and \
                       (RESUME_BOX_TL[1] < hand_y_norm < RESUME_BOX_BR[1]):
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Resuming in {int(MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > MENU_HOLD_DURATION:
                            print("Resuming teleoperation.")
                            robot.move_pose(START_POSE) # Move back to start position
                            is_paused = False # Ensure not paused
                            stop_threads = False # Reset stop flag for new thread
                            control_thread = threading.Thread(target=robot_control_thread, args=(robot,), daemon=True)
                            control_thread.start()
                            current_state = STATE_TELEOP_ACTIVE
                            menu_selection_time = None
                    # Check if hand is in Exit box
                    elif (EXIT_BOX_TL[0] < hand_x_norm < EXIT_BOX_BR[0]) and \
                         (EXIT_BOX_TL[1] < hand_y_norm < EXIT_BOX_BR[1]):
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Exiting in {int(MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > MENU_HOLD_DURATION:
                            print("Exiting program.")
                            current_state = STATE_EXITING # Transition to exit state
                            menu_selection_time = None
                    else:
                        menu_selection_time = None # Reset timer if hand not in any box
                else:
                    menu_selection_time = None # Reset timer if no hand visible
            
            elif current_state == STATE_EXITING:
                # The cleanup in finally block will handle the actual exit
                instruction_text = "Exiting program..."
                stop_threads = True # Ensure cleanup handles thread
                break # Exit the main loop to trigger finally block

            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking Control", frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                stop_threads = True # Ensure threads stop if ESC is pressed
                break
            
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("--- Cleaning up ---")
        stop_threads = True # Ensure threads are signalled to stop
        if control_thread is not None and control_thread.is_alive():
            control_thread.join(timeout=2.0)
        if robot:
            robot.move_pose(HOME_POSE)
            robot.go_to_sleep()
            robot.close_connection()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released.")

if __name__ == "__main__":
    main()