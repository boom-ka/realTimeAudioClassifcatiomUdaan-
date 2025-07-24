import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
import threading
import joblib
import torch
from df import init_df, enhance
import librosa
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
import time

# Load ML resources
mlp_model = joblib.load("mlp_emotion_classifier_best_model2.joblib")
encoder = joblib.load("emotion_encoder.joblib")
scaler = joblib.load("feature_scaler.joblib")

# DeepFilterNet
model, df_state, _ = init_df()

# Constants
SAMPLE_RATE = 16000
BLOCKSIZE = 16000
DENOISE_BLEND = 1.0
SILENCE_THRESHOLD = 0.015

# Buffers
audio_buffer = np.zeros(BLOCKSIZE)
latest_cleaned_audio = np.zeros(BLOCKSIZE)
emotion_label = "Neutral"
confidence_score = 0.0

def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def preprocess(audio):
    return normalize(bandpass_filter(audio))

def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = enhance(model, df_state, tensor_audio.cpu())
    return output.numpy().flatten()

def is_silence(audio):
    return np.sqrt(np.mean(audio ** 2)) < SILENCE_THRESHOLD

def extract_features_from_array(y, sr=16000):
    features = np.array([])

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    features = np.hstack((features, mfccs))

    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, chroma))

    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.hstack((features, mel))

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, contrast))

    y_harm = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr).T, axis=0)
    features = np.hstack((features, tonnetz))

    return features

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_cleaned_audio, emotion_label, confidence_score
    audio_buffer = indata[:, 0]
    processed = preprocess(audio_buffer)
    cleaned = denoise(processed) if not is_silence(processed) else np.zeros_like(processed)
    latest_cleaned_audio = DENOISE_BLEND * cleaned + (1 - DENOISE_BLEND) * processed

    try:
        features = extract_features_from_array(latest_cleaned_audio)
        scaled = scaler.transform([features])
        probs = mlp_model.predict_proba(scaled)[0]
        predicted_idx = np.argmax(probs)
        emotion_label = encoder.inverse_transform([predicted_idx])[0]
        confidence_score = probs[predicted_idx]
    except Exception as e:
        print("[Emotion Error]", e)

class AudioVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Emotion Detection")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Emotion: ...")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.graphs = []
        for _ in range(2):  # Waveform and FFT (raw + clean)
            plot = pg.PlotWidget()
            self.graphs.append(plot)
            self.layout.addWidget(plot)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graphs)
        self.timer.start(50)

    def update_graphs(self):
        self.label.setText(f"Emotion: {emotion_label} | Confidence: {confidence_score:.2f}")

        self.graphs[0].clear()
        self.graphs[0].plot(audio_buffer, pen='r', name="Raw")
        self.graphs[0].plot(latest_cleaned_audio, pen='g', name="Cleaned")
    
        self.graphs[1].clear()
        fft_raw = np.abs(rfft(audio_buffer))
        freqs = rfftfreq(len(audio_buffer), d=1/SAMPLE_RATE)
        self.graphs[1].plot(freqs, fft_raw, pen='r')

        fft_clean = np.abs(rfft(latest_cleaned_audio))
        self.graphs[1].plot(freqs, fft_clean, pen='g')

def start_audio_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        threading.Event().wait()  # Keep stream alive

if __name__ == "__main__":
    # Start audio stream in background thread
    threading.Thread(target=start_audio_stream, daemon=True).start()
    
    # Start Qt GUI in main thread
    app = QApplication(sys.argv)
    window = AudioVisualizer()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
