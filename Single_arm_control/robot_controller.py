# robot_controller.py

import threading
import time
import math
from pyniryo import NiryoRobot
import config
from hand_tracking import scale_value

class RobotController:
    def __init__(self, ip_address):
        self.robot = NiryoRobot(ip_address)
        self.latest_target_pose = None
        self.latest_gesture = None
        self.is_paused = False
        self.stop_threads = False
        self.current_robot_pose = config.START_POSE.copy()
        
        self.pose_lock = threading.Lock()
        self.pose_update_lock = threading.Lock()
        
        # We define the thread here, but it will be re-created upon starting
        self.control_thread = None

    def connect(self):
        self.robot.calibrate_auto()
        self.robot.update_tool()
        print("Robot connected and calibrated.")

    def start_control(self):
        self.stop_threads = False
        # ### FIXED ###: Re-create the thread object before starting.
        # This allows the control loop to be restarted after being stopped.
        if self.control_thread is None or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            print("Robot control thread started.")

    def stop_control(self):
        self.stop_threads = True
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        print("Robot control thread stopped.")

    def update_target(self, target_pose, gesture):
        with self.pose_lock:
            self.latest_target_pose = target_pose
            self.latest_gesture = gesture
    
    def move_to_pose(self, pose):
        self.robot.move_pose(pose)
        with self.pose_update_lock:
            self.current_robot_pose = pose.copy()

    def go_to_sleep(self):
        self.robot.go_to_sleep()
        self.robot.close_connection()
        print("Robot connection closed.")

    def _control_loop(self):
        smoothed_x, smoothed_y, smoothed_z = None, None, None
        last_sent_gesture = None
        
        MODE_ARM_CONTROL = "ARM"
        MODE_GRIPPER_CONTROL = "GRIPPER"
        current_mode = MODE_ARM_CONTROL
        last_toggle_time = 0
        last_rotation_angle = 0.0

        # Set initial smoothed position to the starting pose to avoid a jump
        with self.pose_update_lock:
            smoothed_x = self.current_robot_pose[0]
            smoothed_y = self.current_robot_pose[1]
            smoothed_z = self.current_robot_pose[2]

        while not self.stop_threads:
            if self.is_paused:
                time.sleep(config.COMMAND_INTERVAL)
                continue

            with self.pose_lock:
                target_pose_raw = self.latest_target_pose
                current_gesture = self.latest_gesture
            
            if current_gesture == "Pointing" and (time.time() - last_toggle_time > 1.5):
                current_mode = MODE_GRIPPER_CONTROL if current_mode == MODE_ARM_CONTROL else MODE_ARM_CONTROL
                print(f"Switched to {current_mode} mode")
                last_toggle_time = time.time()
            
            if current_mode == MODE_ARM_CONTROL and target_pose_raw:
                raw_x, raw_y, raw_z = target_pose_raw
                
                # Apply smoothing
                smoothed_x = (config.SMOOTHING_FACTOR * raw_x) + ((1 - config.SMOOTHING_FACTOR) * smoothed_x)
                # ### FIXED ###: Added the missing smoothing logic for Y and Z axes.
                smoothed_y = (config.SMOOTHING_FACTOR * raw_y) + ((1 - config.SMOOTHING_FACTOR) * smoothed_y)
                smoothed_z = (config.SMOOTHING_FACTOR * raw_z) + ((1 - config.SMOOTHING_FACTOR) * smoothed_z)
                
                try:
                    new_pose = [smoothed_x, smoothed_y, smoothed_z, 0.0, 1.57, last_rotation_angle]
                    self.move_to_pose(new_pose)
                except Exception as e:
                    print(f"Robot movement error: {e}")

            elif current_mode == MODE_GRIPPER_CONTROL:
                if current_gesture and current_gesture != last_sent_gesture:
                    if current_gesture in ["Open Hand", "Closed Hand"]:
                        try:
                            if current_gesture == "Open Hand": self.robot.open_gripper(speed=500)
                            else: self.robot.close_gripper(speed=500)
                            last_sent_gesture = current_gesture
                        except Exception as e:
                            print(f"Gripper command error: {e}")
                
                if target_pose_raw:
                    _, hand_y_norm, _ = target_pose_raw
                    try:
                        rotation_angle = scale_value(hand_y_norm, config.ROBOT_MIN_Y, config.ROBOT_MAX_Y, -math.pi/2, math.pi/2)
                        with self.pose_update_lock:
                            rotation_pose = self.current_robot_pose.copy()
                        rotation_pose[5] = rotation_angle
                        self.move_to_pose(rotation_pose)
                        last_rotation_angle = rotation_angle
                    except Exception as e:
                        print(f"Gripper rotation error: {e}")

            time.sleep(config.COMMAND_INTERVAL)