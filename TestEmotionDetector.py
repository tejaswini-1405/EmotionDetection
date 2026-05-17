import cv2
import numpy as np
import os
from keras.models import model_from_json  # type: ignore

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load model
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "emotion_model.json")
h5_path   = os.path.join(base_dir, "emotion_model.h5")

with open(json_path, 'r') as f:
    emotion_model = model_from_json(f.read())

emotion_model.load_weights(h5_path)
print("✅ CNN model loaded for live detection")

# Use OpenCV built-in cascade (fixes hardcoded path bug)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def Start(a):
    cap = cv2.VideoCapture(a)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera not detected.")
            break

        frame = cv2.resize(frame, (850, 720))
        face_detector = cv2.CascadeClassifier(cascade_path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 100, 0), 4)
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cropped = cropped.astype("float32") / 255.0

            prediction = emotion_model.predict(cropped, verbose=0)
            label = emotion_dict[int(np.argmax(prediction))]

            cv2.putText(frame, label, (x + 5, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Press Q to quit
        cv2.putText(frame, "Press Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.imshow('EmotiSense — CNN Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
