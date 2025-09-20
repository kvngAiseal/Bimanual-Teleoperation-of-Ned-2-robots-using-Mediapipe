# main.py

import cv2
import time
import math
import config
from hand_tracking import HandTracker, scale_value
from robot_controller import RobotController

def main():
    # --- Initialization ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = HandTracker(max_hands=2)
    
    print("--- Initializing Robot Connections ---")
    robot_left = RobotController(config.ROBOT_IP_ADDRESS_LEFT)
    robot_right = RobotController(config.ROBOT_IP_ADDRESS_RIGHT)
    
    current_state = config.STATE_WAITING_FOR_HAND
    gesture_start_time = None
    menu_selection_time = None
    
    # --- Tutorial Tracking Variables ---
    active_tutorial_robot = None
    active_tutorial_hand_label = ""
    
    try:
        print("--- Connecting to robots ---")
        robot_left.connect()
        robot_right.connect()
        robot_left.move_to_pose(config.HOME_POSE)
        robot_right.move_to_pose(config.HOME_POSE)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            frame_height, frame_width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            
            # Draw GUI boundaries for control zones
            left_boundary_x = int(config.LEFT_HAND_MAX_X * frame_width)
            right_boundary_x = int(config.RIGHT_HAND_MIN_X * frame_width)
            cv2.line(frame, (left_boundary_x, 0), (left_boundary_x, frame_height), (0, 255, 0), 1)
            cv2.line(frame, (right_boundary_x, 0), (right_boundary_x, frame_height), (0, 255, 0), 1)

            frame, hands_data = tracker.process_frame(frame)
            left_hand = hands_data.get("Left")
            right_hand = hands_data.get("Right")
            
            active_hand_data = hands_data.get(active_tutorial_hand_label)
            active_gesture = active_hand_data["gesture"] if active_hand_data else "No Hand"
            
            instruction_text = ""

            # --- State Machine ---
            if current_state == config.STATE_WAITING_FOR_HAND:
                instruction_text = "Place hands in corresponding boxes"
                l_start = (int(config.LEFT_BOX_TL[0] * frame_width), int(config.LEFT_BOX_TL[1] * frame_height))
                l_end = (int(config.LEFT_BOX_BR[0] * frame_width), int(config.LEFT_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, l_start, l_end, (255, 255, 0), 2)
                r_start = (int(config.RIGHT_BOX_TL[0] * frame_width), int(config.RIGHT_BOX_TL[1] * frame_height))
                r_end = (int(config.RIGHT_BOX_BR[0] * frame_width), int(config.RIGHT_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, r_start, r_end, (0, 255, 255), 2)

                if left_hand and right_hand:
                    lw, lt = left_hand["wrist"], left_hand["middle_finger_tip"]
                    rw, rt = right_hand["wrist"], right_hand["middle_finger_tip"]
                    left_in_box = (config.LEFT_BOX_TL[0] < lw.x < config.LEFT_BOX_BR[0] and config.LEFT_BOX_TL[0] < lt.x < config.LEFT_BOX_BR[0])
                    right_in_box = (config.RIGHT_BOX_TL[0] < rw.x < config.RIGHT_BOX_BR[0] and config.RIGHT_BOX_TL[0] < rt.x < config.RIGHT_BOX_BR[0])
                    
                    if left_in_box and right_in_box:
                        if gesture_start_time is None: gesture_start_time = time.time()
                        time_held = time.time() - gesture_start_time
                        instruction_text = f"Hold Steady... {int(config.HOLD_DURATION - time_held)}s"
                        if time_held > config.HOLD_DURATION:
                            print("--- Starting Tutorial for LEFT arm ---")
                            active_tutorial_robot = robot_left
                            active_tutorial_hand_label = "Left"
                            active_tutorial_robot.move_to_pose(config.TUTORIAL_START_POSE)
                            current_state = config.STATE_TEST_MODE_SWITCH
                            gesture_start_time = None
                    else:
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_MODE_SWITCH:
                instruction_text = f"Raise {active_tutorial_hand_label.upper()} Index Finger"
                if active_gesture == "Pointing":
                    current_state = config.STATE_TEST_GRIPPER_OPEN

            elif current_state == config.STATE_TEST_GRIPPER_OPEN:
                instruction_text = f"{active_tutorial_hand_label.upper()} arm: OPEN HAND"
                if active_gesture == "Open Hand":
                    active_tutorial_robot.robot.open_gripper(speed=500)
                    current_state = config.STATE_TEST_GRIPPER_CLOSE

            elif current_state == config.STATE_TEST_GRIPPER_CLOSE:
                instruction_text = f"{active_tutorial_hand_label.upper()} arm: Make a FIST"
                if active_gesture == "Closed Hand":
                    active_tutorial_robot.robot.close_gripper(speed=500)
                    current_state = config.STATE_TEST_GRIPPER_ROTATION

            elif current_state == config.STATE_TEST_GRIPPER_ROTATION:
                instruction_text = f"{active_tutorial_hand_label.upper()} Hand: Move LEFT & RIGHT"
                if active_hand_data:
                    wrist_x = active_hand_data["wrist"].x
                    min_x, max_x = (config.LEFT_HAND_MIN_X, config.LEFT_HAND_MAX_X) if active_tutorial_hand_label == "Left" else (config.RIGHT_HAND_MIN_X, config.RIGHT_HAND_MAX_X)
                    rotation = scale_value(wrist_x, min_x, max_x, -90, 90)
                    test_pose = active_tutorial_robot.current_robot_pose.copy()
                    test_pose[5] = math.radians(rotation)
                    active_tutorial_robot.move_to_pose(test_pose)
                    if gesture_start_time is None: gesture_start_time = time.time()
                    if time.time() - gesture_start_time > 5.0:
                        current_state = config.STATE_TEST_HOME
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_HOME:
                instruction_text = f"{active_tutorial_hand_label.upper()} Hand: Hold THUMBS UP"
                if active_gesture == "Thumbs Up":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Going Home in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        active_tutorial_robot.move_to_pose(config.HOME_POSE)
                        time.sleep(1)
                        active_tutorial_robot.move_to_pose(config.TUTORIAL_START_POSE)
                        current_state = config.STATE_TEST_PAUSE
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_PAUSE:
                instruction_text = f"{active_tutorial_hand_label.upper()} Hand: Hold PEACE SIGN"
                if active_gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Pausing in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        current_state = config.STATE_TEST_RESUME
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_RESUME:
                instruction_text = f"{active_tutorial_hand_label.upper()} Hand: Hold PEACE again"
                if active_gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Resuming in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        print(f"Tutorial for {active_tutorial_hand_label} arm complete.")
                        active_tutorial_robot.move_to_pose(config.HOME_POSE)
                        if active_tutorial_robot is robot_left:
                            print("--- Starting Tutorial for RIGHT arm ---")
                            active_tutorial_robot = robot_right
                            active_tutorial_hand_label = "Right"
                            active_tutorial_robot.move_to_pose(config.TUTORIAL_START_POSE)
                            current_state = config.STATE_TEST_MODE_SWITCH
                        else:
                            print("--- Tutorial Complete! Starting Teleoperation ---")
                            robot_left.move_to_pose(config.START_POSE)
                            robot_right.move_to_pose(config.RIGHT_ARM_START_POSE)
                            robot_left.start_control()
                            robot_right.start_control()
                            current_state = config.STATE_TELEOP_ACTIVE
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TELEOP_ACTIVE:
                peace_gesture = (left_hand and left_hand["gesture"] == "Peace") or (right_hand and right_hand["gesture"] == "Peace")
                thumbs_up_gesture = (left_hand and left_hand["gesture"] == "Thumbs Up") or (right_hand and right_hand["gesture"] == "Thumbs Up")

                if peace_gesture:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    paused = robot_left.is_paused
                    instruction_text = f"Resuming in {int(config.GESTURE_HOLD_DURATION - time_held)}s" if paused else f"Pausing in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        robot_left.is_paused = not paused
                        robot_right.is_paused = not paused
                        print(f"Teleoperation {'PAUSED' if not paused else 'RESUMED'}")
                        gesture_start_time = None
                elif thumbs_up_gesture and not robot_left.is_paused:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Parking Robots in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        robot_left.stop_control()
                        robot_right.stop_control()
                        robot_left.move_to_pose(config.HOME_POSE)
                        robot_right.move_to_pose(config.HOME_POSE)
                        current_state = config.STATE_CHOICE_MENU
                        gesture_start_time = None
                else:
                    gesture_start_time = None

                if not robot_left.is_paused and left_hand:
                    raw_x = scale_value(left_hand["hand_size"], config.HAND_MAX_SIZE, config.HAND_MIN_SIZE, config.ROBOT_MAX_X, config.ROBOT_MIN_X)
                    raw_y = scale_value(left_hand["wrist"].x, config.LEFT_HAND_MIN_X, config.LEFT_HAND_MAX_X, config.ROBOT_MAX_Y, config.ROBOT_MIN_Y)
                    raw_z = scale_value(left_hand["wrist"].y, config.HAND_TRACKING_MIN_Y, config.HAND_TRACKING_MAX_Y, config.ROBOT_MAX_Z, config.ROBOT_MIN_Z)
                    robot_left.update_target([raw_x, raw_y, raw_z], left_hand["gesture"])
                else:
                    robot_left.update_target(None, "No Hand")

                if not robot_right.is_paused and right_hand:
                    raw_x = scale_value(right_hand["hand_size"], config.HAND_MAX_SIZE, config.HAND_MIN_SIZE, config.ROBOT_MAX_X, config.ROBOT_MIN_X)
                    raw_y = scale_value(right_hand["wrist"].x, config.RIGHT_HAND_MIN_X, config.RIGHT_HAND_MAX_X, config.ROBOT_MAX_Y, config.ROBOT_MIN_Y)
                    raw_z = scale_value(right_hand["wrist"].y, config.HAND_TRACKING_MIN_Y, config.HAND_TRACKING_MAX_Y, config.ROBOT_MAX_Z, config.ROBOT_MIN_Z)
                    robot_right.update_target([raw_x, raw_y, raw_z], right_hand["gesture"])
                else:
                    robot_right.update_target(None, "No Hand")

                if gesture_start_time is None:
                    instruction_text = "Bimanual Teleoperation Active"

            elif current_state == config.STATE_CHOICE_MENU:
                instruction_text = "Choose an Option"
                resume_start = (int(config.RESUME_BOX_TL[0] * frame_width), int(config.RESUME_BOX_TL[1] * frame_height))
                resume_end = (int(config.RESUME_BOX_BR[0] * frame_width), int(config.RESUME_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, resume_start, resume_end, (0, 255, 255), 2)
                cv2.putText(frame, "Resume", (resume_start[0] + 10, resume_start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                exit_start = (int(config.EXIT_BOX_TL[0] * frame_width), int(config.EXIT_BOX_TL[1] * frame_height))
                exit_end = (int(config.EXIT_BOX_BR[0] * frame_width), int(config.EXIT_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, exit_start, exit_end, (0, 0, 255), 2)
                cv2.putText(frame, "Exit", (exit_start[0] + 10, exit_start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                menu_hand = left_hand or right_hand
                if menu_hand:
                    hx, hy = menu_hand["wrist"].x, menu_hand["wrist"].y
                    if config.RESUME_BOX_TL[0] < hx < config.RESUME_BOX_BR[0] and config.RESUME_BOX_TL[1] < hy < config.RESUME_BOX_BR[1]:
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Resuming in {int(config.MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > config.MENU_HOLD_DURATION:
                            robot_left.move_to_pose(config.START_POSE)
                            robot_right.move_to_pose(config.RIGHT_ARM_START_POSE)
                            robot_left.start_control()
                            robot_right.start_control()
                            current_state = config.STATE_TELEOP_ACTIVE
                            menu_selection_time = None
                    elif config.EXIT_BOX_TL[0] < hx < config.EXIT_BOX_BR[0] and config.EXIT_BOX_TL[1] < hy < config.EXIT_BOX_BR[1]:
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Exiting in {int(config.MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > config.MENU_HOLD_DURATION:
                            current_state = config.STATE_EXITING
                            menu_selection_time = None
                    else:
                        menu_selection_time = None
                else:
                    menu_selection_time = None
            
            elif current_state == config.STATE_EXITING:
                instruction_text = "Exiting program..."
                break

            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Bimanual Hand Tracking Control", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed. Exiting.")
                break
                
    except Exception as e:
        print(f"A critical error occurred: {e}")

    finally:
        print("--- Cleaning up ---")
        robot_left.stop_control()
        robot_right.stop_control()
        try:
            print("Parking robots...")
            robot_left.move_to_pose(config.HOME_POSE)
            robot_right.move_to_pose(config.HOME_POSE)
            robot_left.go_to_sleep()
            robot_right.go_to_sleep()
        except Exception as e:
            print(f"Could not park or sleep robots during cleanup: {e}")
        
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main()