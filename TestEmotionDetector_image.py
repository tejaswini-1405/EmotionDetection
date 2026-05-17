import cv2
import numpy as np
import os

# ─── Emotion labels ───────────────────────────────────────────
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ─── Module-level state ───────────────────────────────────────
_last_emotion = "Neutral"

def get_last_emotion():
    """Return the emotion detected in the most recent File() call."""
    return _last_emotion

# ─── Load model (lazy — avoids crash on import if files missing) ──
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model

    from keras.models import model_from_json  # type: ignore

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "emotion_model.json")
    h5_path   = os.path.join(base_dir, "emotion_model.h5")

    with open(json_path, "r") as f:
        _model = model_from_json(f.read())

    _model.load_weights(h5_path)
    print("✅ CNN emotion model loaded")
    return _model


# ─── Image emotion detection ──────────────────────────────────
def File(input_path):
    """
    Run CNN emotion detection on an image file.
    Saves annotated result to static/test1/output.jpg
    Returns output_path.
    """
    global _last_emotion

    model = _load_model()

    output_dir = "static/test1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.jpg")

    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Cannot load image: {input_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Always use OpenCV's built-in cascade path (fixes hardcoded path bug)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print(f"CNN — Faces detected: {len(faces)}")

    if len(faces) == 0:
        cv2.putText(
            frame, "No Face Detected", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
        )
        _last_emotion = "Neutral"
        cv2.imwrite(output_path, frame)
        return output_path

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)

        # Preprocess ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict
        prediction = model.predict(roi_gray, verbose=0)
        label = emotion_dict[int(np.argmax(prediction))]
        _last_emotion = label

        # Label with background box
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), (255, 100, 0), -1)
        cv2.putText(
            frame, label, (x + 5, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )

    cv2.imwrite(output_path, frame)
    print(f"✅ CNN output saved: {output_path} | Emotion: {_last_emotion}")
    return output_path
