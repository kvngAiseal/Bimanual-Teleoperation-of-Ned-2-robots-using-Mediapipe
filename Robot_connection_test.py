from pyniryo import NiryoRobot

# !!! IMPORTANT !!!
# Replace this with your robot's actual IP address.
ROBOT_IP_ADDRESS = "192.168.8.146" 

robot = None
try:
    # --- 1. Connect to the Robot ---
    print(f"Connecting to robot at IP: {ROBOT_IP_ADDRESS}...")
    robot = NiryoRobot(ROBOT_IP_ADDRESS)
    print("Connection successful!")

    # --- 2. Calibrate ---
    # This function should block and wait for completion by itself.
    print("Starting calibration... (This will take a moment)")
    robot.calibrate_auto() 
    print("Calibration complete.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- 3. Close the Connection ---
    if robot:
        print("Closing connection.")
        robot.close_connection()
        print("Connection closed.")