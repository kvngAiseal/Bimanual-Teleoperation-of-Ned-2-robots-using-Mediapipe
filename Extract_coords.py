import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Printing wrist coordinates. Press 'q' to quit.")

# Main loop to capture and process frames
while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- NEW CODE STARTS HERE ---
            
            # Get the coordinates of the wrist (landmark 0)
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Print the normalized coordinates to the terminal
            print(f'Wrist Coords: X={wrist_landmark.x:.2f}, Y={wrist_landmark.y:.2f}')
            
            # --- NEW CODE ENDS HERE ---

    cv2.imshow('MediaPipe Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam feed closed.")