import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

def classify_gesture(landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers_extended = []

    # Thumb: handedness-aware horizontal check (stable for webcam use)
    thumb_tip = landmarks.landmark[tip_ids[0]]
    thumb_ip  = landmarks.landmark[tip_ids[0] - 1]
    if hand_label == "Right":
        thumb_ext = thumb_tip.x < thumb_ip.x
    else:  # Left hand
        thumb_ext = thumb_tip.x > thumb_ip.x
    fingers_extended.append(thumb_ext)

    # Fingers (Index → Pinky): tip above PIP => extended
    for tip_id in tip_ids[1:]:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[tip_id - 2]
        fingers_extended.append(tip.y < pip.y)

    # Patterns
    if all(fingers_extended):
        return "Open Palm"
    if not any(fingers_extended):
        return "Fist"
    if fingers_extended[1] and not any(fingers_extended[i] for i in [0, 2, 3, 4]):
        return "Point"
    if fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[i] for i in [0, 3, 4]):
        return "Peace"
    if fingers_extended[0] and not any(fingers_extended[1:]):
        return "Thumbs Up"
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
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            gesture = classify_gesture(hand_landmarks, label)
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
