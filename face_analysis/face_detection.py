import cv2
from deepface import DeepFace
from collections import deque

def detect_face_emotion():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print("Failed to open the camera. Exiting...")
        return "No camera available"

    max_consecutive_same_emotion = 30  # Frames to detect the same emotion
    emotion_history = deque(maxlen=max_consecutive_same_emotion)
    emotion = "No face detected"  # Default value for emotion

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        # Face detection and emotion analysis
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
            emotion = result[0]['dominant_emotion']
            emotion_history.append(emotion)
        except Exception as e:
            print(f"Error: {e}")
            emotion_history.append("No face detected")

        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)

        # Exit if the same emotion persists for a certain period
        if len(emotion_history) == max_consecutive_same_emotion and all(
            em == emotion_history[0] for em in emotion_history
        ):
            print(f"Detected '{emotion}' continuously. Exiting...")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on pressing 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion
