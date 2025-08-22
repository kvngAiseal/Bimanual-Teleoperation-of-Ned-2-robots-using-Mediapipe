import cv2
import mediapipe as mp
from pyniryo import NiryoRobot
import time

# ========== CONFIGURATION ==========
ROBOT_IP_ADDRESS = "192.168.8.146"
# ===================================

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

def classify_gesture(hand_landmarks):
    """Classifies a hand as 'Open Hand' or 'Closed Hand'."""
    lm = hand_landmarks.landmark
    fingers_extended = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    return "Open Hand" if fingers_extended else "Closed Hand"

def main():
    cap = cv2.VideoCapture(0)
    robot = None

    try:
        print("--- Connecting to robot ---")
        robot = NiryoRobot(ROBOT_IP_ADDRESS)
        print("Connected successfully.")

        print("--- Updating tool ---")
        robot.update_tool()
        print("Tool updated.")

        prev_gesture = None
        cooldown = 2.0  # seconds
        last_action_time = time.time()

        print("Ready for gesture control. Show your hand to open/close the gripper.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            instruction_text = "Show Hand to Control Gripper"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    gesture = classify_gesture(hand_landmarks)
                    instruction_text = f"Gesture: {gesture}"

                    current_time = time.time()
                    if gesture != prev_gesture and (current_time - last_action_time) > cooldown:
                        try:
                            if gesture == "Open Hand":
                                robot.open_gripper(speed=500)
                            else:
                                robot.close_gripper(speed=500)
                            prev_gesture = gesture
                            last_action_time = current_time
                        except Exception as e:
                            instruction_text = f"Gripper error: {str(e)}"
                            print("Gripper command failed:", e)

            cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gripper Control", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                break

    except Exception as e:
        print("Setup failed:", e)

    finally:
        print("--- Cleaning up ---")
        if robot:
            # robot.go_to_sleep()  # Skip sleep during testing
            robot.close_connection()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released.")

if __name__ == "__main__":
    main()
