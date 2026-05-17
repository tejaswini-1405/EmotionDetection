import cv2
from deepface import DeepFace
import time

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def Start(a):
    cap = cv2.VideoCapture(a)
    prev_time = 0
    emotion_label = "Detecting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera not detected.")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            # Limit DeepFace analysis to once every 1.5 seconds
            if time.time() - prev_time > 1.5:
                try:
                    result = DeepFace.analyze(
                        face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    emotion_label = result[0]['dominant_emotion']
                    prev_time = time.time()
                except Exception as e:
                    print("⚠️ Error:", e)
                    emotion_label = "No face"

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Real-time Emotion Detection (Optimized)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



