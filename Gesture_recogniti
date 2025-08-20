import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Simple gesture logic: Open palm vs. fist
def classify_gesture(landmarks):
    tip_ids = [4, 8, 12, 16, 20]  # Thumb & Finger tips
    count_extended = 0

    for tip_id in tip_ids[1:]:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[tip_id - 2]  # Corresponding PIP joint

        if tip.y < pip.y:  # Finger is extended
            count_extended += 1

    # Thumb logic (optional, simplified)
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    if thumb_tip.x < thumb_ip.x:  # Extended thumb (flipped frame)
        count_extended += 1

    if count_extended >= 4:
        return "Open Palm"
    elif count_extended <= 1:
        return "Fist"
    else:
        return "Unclear Gesture"

# Webcam feed
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "No Hand"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = classify_gesture(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Gesture: {gesture}', (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()