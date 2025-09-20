# main.py

import cv2
import time
import math
import config  # Import your config file
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
    
    tracker = HandTracker()
    robot = RobotController(config.ROBOT_IP_ADDRESS)
    
    current_state = config.STATE_WAITING_FOR_HAND
    gesture_start_time = None
    menu_selection_time = None
    
    try:
        print("--- Connecting to robot ---")
        robot.connect()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("Error: Failed to grab frame.")
                break
            
            frame_height, frame_width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            frame, hand_data = tracker.process_frame(frame)
            gesture = hand_data["gesture"] if hand_data else "No Hand"
            
            instruction_text = ""

            # --- State Machine Logic ---
            
            if current_state == config.STATE_WAITING_FOR_HAND:
                instruction_text = "Place Hand in Box to Start"
                start_point = (int(config.BOX_TOP_LEFT[0] * frame_width), int(config.BOX_TOP_LEFT[1] * frame_height))
                end_point = (int(config.BOX_BOTTOM_RIGHT[0] * frame_width), int(config.BOX_BOTTOM_RIGHT[1] * frame_height))
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
                
                if hand_data:
                    wrist_x = hand_data["wrist"].x
                    wrist_y = hand_data["wrist"].y
                    if (config.BOX_TOP_LEFT[0] < wrist_x < config.BOX_BOTTOM_RIGHT[0]) and \
                       (config.BOX_TOP_LEFT[1] < wrist_y < config.BOX_BOTTOM_RIGHT[1]):
                        if gesture_start_time is None: gesture_start_time = time.time()
                        time_held = time.time() - gesture_start_time
                        instruction_text = f"Hold Steady... {int(config.HOLD_DURATION - time_held)}s"
                        if time_held > config.HOLD_DURATION:
                            print("Step 1 Complete: Moving to start pose.")
                            robot.move_to_pose(config.START_POSE)
                            current_state = config.STATE_TEST_MODE_SWITCH
                            gesture_start_time = None
                    else:
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_MODE_SWITCH:
                instruction_text = "Raise Index Finger to test Mode Switch"
                if gesture == "Pointing":
                    print("Step 2 Complete: Mode switch tested.")
                    current_state = config.STATE_TEST_GRIPPER_OPEN
            
            elif current_state == config.STATE_TEST_GRIPPER_OPEN:
                instruction_text = "OPEN HAND to Open Gripper"
                if gesture == "Open Hand":
                    robot.robot.open_gripper(speed=500)
                    print("Step 3 Complete: Gripper open tested.")
                    current_state = config.STATE_TEST_GRIPPER_CLOSE
            
            elif current_state == config.STATE_TEST_GRIPPER_CLOSE:
                instruction_text = "Make a FIST to Close Gripper"
                if gesture == "Closed Hand":
                    robot.robot.close_gripper(speed=500)
                    print("Step 4 Complete: Gripper close tested.")
                    current_state = config.STATE_TEST_GRIPPER_ROTATION
            
            elif current_state == config.STATE_TEST_GRIPPER_ROTATION:
                instruction_text = "Move hand LEFT and RIGHT to rotate gripper"
                if hand_data:
                    test_rotation = scale_value(hand_data["wrist"].x, 0.2, 0.8, -90, 90)
                    cv2.putText(frame, f"Rotation: {int(test_rotation)} degrees", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    try:
                        rotation_angle_rad = math.radians(test_rotation)
                        with robot.pose_update_lock:
                            test_pose = robot.current_robot_pose.copy()
                        test_pose[5] = rotation_angle_rad
                        robot.move_to_pose(test_pose)
                    except Exception as e:
                        print(f"Test rotation error: {e}")
                    
                    if gesture_start_time is None: gesture_start_time = time.time()
                    if time.time() - gesture_start_time > 5.0:
                        print("Step 5 Complete: Gripper rotation tested.")
                        reset_pose = robot.current_robot_pose.copy()
                        reset_pose[5] = 0.0
                        robot.move_to_pose(reset_pose)
                        current_state = config.STATE_TEST_HOME
                        gesture_start_time = None
                else:
                    gesture_start_time = None
            
            elif current_state == config.STATE_TEST_HOME:
                instruction_text = "Hold THUMBS UP to test Go Home"
                if gesture == "Thumbs Up":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Going Home in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        robot.move_to_pose(config.HOME_POSE)
                        time.sleep(1)
                        robot.move_to_pose(config.START_POSE)
                        print("Step 6 Complete: Go Home tested.")
                        current_state = config.STATE_TEST_PAUSE
                        gesture_start_time = None
                else:
                    gesture_start_time = None
            
            elif current_state == config.STATE_TEST_PAUSE:
                instruction_text = "Hold PEACE SIGN to test Pause"
                if gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Pausing in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        print("Step 7 Complete: Pause tested.")
                        current_state = config.STATE_TEST_RESUME
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TEST_RESUME:
                instruction_text = "Hold PEACE SIGN again to test Resume"
                if gesture == "Peace":
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Resuming in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        print("Tutorial Complete! Starting teleoperation.")
                        robot.start_control()
                        current_state = config.STATE_TELEOP_ACTIVE
                        gesture_start_time = None
                else:
                    gesture_start_time = None

            elif current_state == config.STATE_TELEOP_ACTIVE:
                is_holding_peace = (gesture == "Peace")
                is_holding_thumbs_up = (gesture == "Thumbs Up")
                
                if is_holding_peace:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Resuming in {int(config.GESTURE_HOLD_DURATION - time_held)}s" if robot.is_paused else f"Pausing in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        robot.is_paused = not robot.is_paused
                        print(f"Teleoperation {'PAUSED' if robot.is_paused else 'RESUMED'}")
                        gesture_start_time = None
                elif is_holding_thumbs_up and not robot.is_paused:
                    if gesture_start_time is None: gesture_start_time = time.time()
                    time_held = time.time() - gesture_start_time
                    instruction_text = f"Parking Robot in {int(config.GESTURE_HOLD_DURATION - time_held)}s"
                    if time_held > config.GESTURE_HOLD_DURATION:
                        print("Parking robot and showing menu.")
                        robot.stop_control()
                        robot.move_to_pose(config.HOME_POSE)
                        current_state = config.STATE_CHOICE_MENU
                        gesture_start_time = None
                else:
                    gesture_start_time = None
                
                if gesture_start_time is None:
                    instruction_text = "PAUSED (Hold Peace to Resume)" if robot.is_paused else "Teleoperation Active (Point=Toggle Mode)"

                if not robot.is_paused and hand_data:
                    raw_x = scale_value(hand_data["hand_size"], config.HAND_MAX_SIZE, config.HAND_MIN_SIZE, config.ROBOT_MAX_X, config.ROBOT_MIN_X)
                    raw_y = scale_value(hand_data["wrist"].x, config.HAND_TRACKING_MIN_X, config.HAND_TRACKING_MAX_X, config.ROBOT_MIN_Y, config.ROBOT_MAX_Y)
                    raw_z = scale_value(hand_data["wrist"].y, config.HAND_TRACKING_MIN_Y, config.HAND_TRACKING_MAX_Y, config.ROBOT_MAX_Z, config.ROBOT_MIN_Z)
                    robot.update_target([raw_x, raw_y, raw_z], gesture)
                else:
                    robot.update_target(None, gesture)
            
            elif current_state == config.STATE_CHOICE_MENU:
                instruction_text = "Choose an Option"
                # Draw Resume box
                resume_box_start = (int(config.RESUME_BOX_TL[0] * frame_width), int(config.RESUME_BOX_TL[1] * frame_height))
                resume_box_end = (int(config.RESUME_BOX_BR[0] * frame_width), int(config.RESUME_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, resume_box_start, resume_box_end, (0, 255, 255), 2)
                cv2.putText(frame, "Resume", (resume_box_start[0] + 10, resume_box_start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw Exit box
                exit_box_start = (int(config.EXIT_BOX_TL[0] * frame_width), int(config.EXIT_BOX_TL[1] * frame_height))
                exit_box_end = (int(config.EXIT_BOX_BR[0] * frame_width), int(config.EXIT_BOX_BR[1] * frame_height))
                cv2.rectangle(frame, exit_box_start, exit_box_end, (0, 0, 255), 2)
                cv2.putText(frame, "Exit", (exit_box_start[0] + 10, exit_box_start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if hand_data:
                    wrist_x = hand_data["wrist"].x
                    wrist_y = hand_data["wrist"].y
                    
                    in_resume_box = (config.RESUME_BOX_TL[0] < wrist_x < config.RESUME_BOX_BR[0]) and (config.RESUME_BOX_TL[1] < wrist_y < config.RESUME_BOX_BR[1])
                    in_exit_box = (config.EXIT_BOX_TL[0] < wrist_x < config.EXIT_BOX_BR[0]) and (config.EXIT_BOX_TL[1] < wrist_y < config.EXIT_BOX_BR[1])

                    if in_resume_box:
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Resuming in {int(config.MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > config.MENU_HOLD_DURATION:
                            print("Resuming teleoperation.")
                            robot.move_to_pose(config.START_POSE)
                            robot.is_paused = False
                            robot.start_control()
                            current_state = config.STATE_TELEOP_ACTIVE
                            menu_selection_time = None
                    elif in_exit_box:
                        if menu_selection_time is None: menu_selection_time = time.time()
                        time_in_box = time.time() - menu_selection_time
                        instruction_text = f"Exiting in {int(config.MENU_HOLD_DURATION - time_in_box)}s"
                        if time_in_box > config.MENU_HOLD_DURATION:
                            print("Exiting program.")
                            current_state = config.STATE_EXITING
                            menu_selection_time = None
                    else:
                        menu_selection_time = None
                else:
                    menu_selection_time = None

            elif current_state == config.STATE_EXITING:
                instruction_text = "Exiting program..."
                break

            # Display UI
            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking Control", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                print("ESC pressed. Exiting.")
                break
                
    except Exception as e:
        print(f"A critical error occurred in main loop: {e}")

    finally:
        print("--- Cleaning up ---")
        robot.stop_control()
        # Final safety check to ensure robot is parked
        try:
            robot.move_to_pose(config.HOME_POSE)
            robot.go_to_sleep()
        except Exception as e:
            print(f"Could not park or sleep robot during cleanup: {e}")
        
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main()