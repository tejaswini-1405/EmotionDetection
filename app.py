from flask import Flask, request, render_template
import os

from emotion_image import File as DeepFaceFile
from TestEmotionDetector_image import File as CNNFile, get_last_emotion as CNNEmotion

app = Flask(__name__)

# ─── Emoji map ────────────────────────────────────────────────
EMOJI_MAP = {
    "angry":     "😠",
    "disgusted": "🤢",
    "fearful":   "😨",
    "fear":      "😨",
    "happy":     "😄",
    "neutral":   "😐",
    "sad":       "😢",
    "surprised": "😲",
    "surprise":  "😲",
    "Angry":     "😠",
    "Disgusted": "🤢",
    "Fearful":   "😨",
    "Happy":     "😄",
    "Neutral":   "😐",
    "Sad":       "😢",
    "Surprised": "😲",
}

def emotion_emoji(emotion):
    if not emotion:
        return "🔍"
    return EMOJI_MAP.get(emotion, "🔍")


# ─── HOME ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─── DEEPFACE IMAGE ────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
    file = request.files.get('file')
    if not file or file.filename == "":
        return render_template('index.html', error="No file selected")

    upload_folder = "static/test"
    os.makedirs(upload_folder, exist_ok=True)
    input_path = os.path.join(upload_folder, file.filename)
    file.save(input_path)

    try:
        output_path, emotion = DeepFaceFile(input_path)
    except Exception as e:
        return render_template('index.html', error=f"DeepFace error: {str(e)}")

    return render_template(
        'index.html',
        image1=input_path,
        image2=output_path,
        emotion=emotion,
        emoji=emotion_emoji(emotion)
    )


# ─── DEEPFACE LIVE ─────────────────────────────────────────────
@app.route('/Live')
def Live():
    from emotion import Start
    Start(0)
    return render_template('index.html')


# ─── CNN IMAGE ─────────────────────────────────────────────────
@app.route('/analyse1', methods=['POST'])
def analyse1():
    file = request.files.get('file')
    if not file or file.filename == "":
        return render_template('index.html', error="No file selected")

    upload_folder = "static/test1"
    os.makedirs(upload_folder, exist_ok=True)
    input_path = os.path.join(upload_folder, file.filename)
    file.save(input_path)

    try:
        output_path = CNNFile(input_path)
        emotion_cnn = CNNEmotion()
    except Exception as e:
        return render_template('index.html', error=f"CNN error: {str(e)}")

    return render_template(
        'index.html',
        image3=input_path,
        image4=output_path,
        emotion_cnn=emotion_cnn,
        emoji_cnn=emotion_emoji(emotion_cnn)
    )


# ─── CNN LIVE ──────────────────────────────────────────────────
@app.route('/Live11')
def Live11():
    from TestEmotionDetector import Start
    Start(0)
    return render_template('index.html')


# ─── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
