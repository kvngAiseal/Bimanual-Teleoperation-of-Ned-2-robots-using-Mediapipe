# hand_tracking.py

import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        """Processes a single frame to find multiple hands and their data."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_data = {}
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i]
                hand_label = handedness.classification[0].label

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                gesture = self.classify_gesture(hand_landmarks, hand_label)
                
                hands_data[hand_label] = {
                    "landmarks": hand_landmarks,
                    "gesture": gesture,
                    "wrist": hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                    # ### ADDED ### Pass the middle finger tip landmark for startup check
                    "middle_finger_tip": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    "hand_size": self.get_hand_size(hand_landmarks)
                }
                
        return frame, hands_data

    def get_hand_size(self, hand_landmarks):
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        return math.sqrt((wrist.x - mcp_middle.x)**2 + (wrist.y - mcp_middle.y)**2)

    def classify_gesture(self, landmarks, hand_label):
        tip_ids = [4, 8, 12, 16, 20]
        fingers_extended = []

        # Thumb
        thumb_tip = landmarks.landmark[tip_ids[0]]
        thumb_ip = landmarks.landmark[tip_ids[0] - 1]
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

        if fingers_extended[0] and not any(fingers_extended[1:]):
            return "Thumbs Up"
        if fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[i] for i in [0, 3, 4]):
            return "Peace"
        if fingers_extended[1] and not any(fingers_extended[i] for i in [0, 2, 3, 4]):
            return "Pointing"
        if all(fingers_extended):
            return "Open Hand"
        if not any(fingers_extended):
            return "Closed Hand"
        return "Unclear"

    def close(self):
        self.hands.close()

def scale_value(value, in_min, in_max, out_min, out_max):
    """ Scales a value from one range to another, handling inverted ranges. """
    if in_min > in_max:
        in_min, in_max = in_max, in_min
        value = max(in_min, min(value, in_max))
        in_min, in_max = in_max, in_min
    else:
        value = max(in_min, min(value, in_max))

    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min