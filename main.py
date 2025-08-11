from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import noisereduce as nr

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('sentiment_cnn_model.h5')
le = pickle.load(open('le.pkl','rb'))

# Audio upload path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Improved feature extraction
def extract_features(file_name):
    # Load audio (mono, fixed sample rate)
    audio, sample_rate = librosa.load(file_name, sr=22050, res_type='kaiser_fast')

    # Noise reduction
    audio = nr.reduce_noise(y=audio, sr=sample_rate)

    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Normalize volume
    audio = librosa.util.normalize(audio)

    # Ensure fixed length (e.g., 3 seconds)
    desired_length = sample_rate * 3
    if len(audio) < desired_length:
        audio = np.pad(audio, (0, desired_length - len(audio)))
    else:
        audio = audio[:desired_length]

    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    return np.mean(mfccs.T, axis=0)

# Prediction
def predict_sentiment(audio_path):
    feature = extract_features(audio_path)
    feature = feature.reshape(1, 40, 1, 1)
    prediction = model.predict(feature)
    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["audio"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            predicted_sentiment = predict_sentiment(file_path)

            return render_template(
                "index.html",
                sentiment=predicted_sentiment,
                audio_path=filename  # send only filename, not full path
            )
    return render_template("index.html", sentiment=None, audio_path=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)