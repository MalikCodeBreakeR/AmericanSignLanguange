import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition function for ASL
def recognize_gesture(hand_landmarks):
    # Extract the landmark positions
    landmarks = {}
    for id, lm in enumerate(hand_landmarks.landmark):
        landmarks[id] = (lm.x, lm.y)

    # Define gestures based on landmark positions for ASL A-Z
    # Example conditions for some letters
    if landmarks.get(4)[0] < landmarks.get(8)[0] and landmarks.get(0)[1] > landmarks.get(9)[1]:  # A
        return "A"
    # elif landmarks.get(4)[1] < landmarks.get(6)[1] and landmarks.get(8)[0] < landmarks.get(7)[0]:  # B
    #     return "B"
    # elif landmarks.get(4)[1] > landmarks.get(8)[1] and landmarks.get(7)[1] < landmarks.get(5)[1]:  # C
    #     return "C"
    else:
        return 'Nothing'
    # Add similar conditions for D-Z based on ASL gestures

    return None

# Function to start the hand gesture recognition
def hand_gesture_recognition():
    cap = cv2.VideoCapture(1)  # Change the index if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a later selfie-view display
        # frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)
                if gesture:
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hand_gesture_recognition()
