import cv2
from deepface import DeepFace
import os

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def File(input_path):
    """
    Run DeepFace emotion detection on an image.
    Returns (output_path, dominant_emotion) tuple.
    """
    folder = os.path.dirname(input_path)
    output_path = os.path.join(folder, "out.jpg")

    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Cannot load image from: {input_path}")

    # Resize for consistent processing
    frame = cv2.resize(frame, (640, 480))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    dominant_emotion = "neutral"

    if len(faces) == 0:
        cv2.putText(
            frame, "No Face Detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
        )
        cv2.imwrite(output_path, frame)
        return output_path, "neutral"

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            dominant_emotion = result[0]['dominant_emotion']
        except Exception as e:
            print("DeepFace error:", e)
            dominant_emotion = "neutral"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 2)

        # Emotion label with background for readability
        label = dominant_emotion.upper()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), (0, 200, 100), -1)
        cv2.putText(frame, label, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, frame)
    print(f"✅ DeepFace output saved: {output_path} | Emotion: {dominant_emotion}")
    return output_path, dominant_emotion
