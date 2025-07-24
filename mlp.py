from scipy.signal import spectrogram
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, lfilter
import threading
import torch
import wave
import atexit
from df import init_df, enhance
# === Load classifier, encoder, and scaler ===
import joblib
import librosa
import time  # Add at the top if not already


mlp_model = joblib.load("mlp_emotion_classifier_best_model2.joblib")
encoder = joblib.load("emotion_encoder.joblib")
scaler = joblib.load("feature_scaler.joblib")

# === Init DeepFilterNet ===
model, df_state, _ = init_df()

# === Globals ===
buffer_lock = threading.Lock()
SAMPLE_RATE = 16000
BLOCKSIZE = 16000  # 0.5 sec
DENOISE_BLEND = 1  # 0.0 = no denoising, 1.0 = full denoising
SILENCE_THRESHOLD = 0.015  # Tune this if silence detection is too sensitive or insensitive

audio_buffer = np.zeros(BLOCKSIZE)
latest_cleaned_audio = np.zeros(BLOCKSIZE)
raw_audio_storage = []
clean_audio_storage = []

# === Bandpass filter (300–3400 Hz) ===
def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

# === Normalize to [-1, 1] ===
def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

# === Preprocessing before denoising ===
def preprocess(audio):
    filtered = bandpass_filter(audio)
    normalized = normalize(filtered)
    return normalized

# === Denoise without return_mask (DeepFilterNet v0.5.6) ===
def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = enhance(model, df_state, tensor_audio.cpu())
    return output.numpy().flatten()

# === Check if audio is silence based on energy ===
def is_silence(audio, threshold=SILENCE_THRESHOLD):
    energy = np.sqrt(np.mean(audio ** 2))
    return energy < threshold

# === Save audio to WAV ===
def save_audio(filename, data):
    flat = np.concatenate(data)
    scaled = np.int16(flat / (np.max(np.abs(flat)) + 1e-6) * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(scaled.tobytes())

# === Save on exit ===
def on_exit():
    print("\nSaving audio...")
    save_audio("radio_raw.wav", raw_audio_storage)
    save_audio("radio_cleaned.wav", clean_audio_storage)
    print("Done.")

atexit.register(on_exit)

# === Audio callback ===
import time  # Add at the top if not already

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_cleaned_audio
    audio_buffer = indata[:, 0]
    raw_audio_storage.append(audio_buffer.copy())

    # === Preprocessing ===
    processed = preprocess(audio_buffer)

    # === Denoising timing ===
    start_denoise = time.perf_counter()
    cleaned = denoise(processed)
    denoise_time = (time.perf_counter() - start_denoise) * 1000  # ms

    # === Conditional blending based on silence ===
    if is_silence(processed):
        output_audio = np.zeros_like(cleaned)
    else:
        output_audio = DENOISE_BLEND * cleaned + (1 - DENOISE_BLEND) * processed

    latest_cleaned_audio = output_audio
    clean_audio_storage.append(output_audio.copy())

    # === Emotion prediction timing ===
    try:
        start_emotion = time.perf_counter()
        features = extract_features_from_array(output_audio, sr=SAMPLE_RATE)
        features_scaled = scaler.transform([features])
        probs = mlp_model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]
        emotion = encoder.inverse_transform([predicted_idx])[0]
        emotion_time = (time.perf_counter() - start_emotion) * 1000  # ms

        print(f"[Emotion Detected] {emotion} (Confidence: {confidence:.2f}) | "
              f"Denoise: {denoise_time:.2f} ms | Emotion: {emotion_time:.2f} ms")

    except Exception as e:
        print(f"[Emotion Detection Error] {e}")



# === Plotting ===

# from scipy.signal import spectrogram

# from scipy.signal import spectrogram

def plot_live():
    plt.ion()
    fig, axs = plt.subplots(5, figsize=(10, 12))
    fig.suptitle("Real-Time Audio Visualization", fontsize=14)

    while True:
        with buffer_lock:
            buffer_copy = audio_buffer.copy()
            clean_copy = latest_cleaned_audio.copy()

        # --- Plot 1: Time Domain (Raw) ---
        axs[0].cla()
        axs[0].plot(buffer_copy)
        axs[0].set_title("Time Domain (Raw)")
        axs[0].set_ylim([-1, 1])
        axs[0].set_xlabel("Samples")
        axs[0].set_ylabel("Amplitude")

        # --- Plot 2: Spectrogram (Raw) ---
        axs[1].cla()
        f_raw, t_raw, Sxx_raw = spectrogram(buffer_copy, SAMPLE_RATE, nperseg=256, noverlap=128)
        axs[1].imshow(10 * np.log10(Sxx_raw + 1e-10), origin='lower', aspect='auto',
                      extent=[t_raw[0], t_raw[-1], f_raw[0], f_raw[-1]], cmap='viridis')
        axs[1].set_title("Spectrogram (Raw)")
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylim([0, 4000])

        # --- Plot 3: Spectrogram (Cleaned) ---
        axs[2].cla()
        f_clean, t_clean, Sxx_clean = spectrogram(clean_copy, SAMPLE_RATE, nperseg=256, noverlap=128)
        axs[2].imshow(10 * np.log10(Sxx_clean + 1e-10), origin='lower', aspect='auto',
                      extent=[t_clean[0], t_clean[-1], f_clean[0], f_clean[-1]], cmap='viridis')
        axs[2].set_title("Spectrogram (Cleaned)")
        axs[2].set_ylabel("Frequency (Hz)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylim([0, 4000])

        # --- Plot 4: Frequency Domain (Raw) ---
        axs[3].cla()
        fft_raw = np.abs(np.fft.rfft(buffer_copy))
        freqs_raw = np.fft.rfftfreq(len(buffer_copy), d=1/SAMPLE_RATE)
        axs[3].plot(freqs_raw, fft_raw)
        axs[3].set_title("Frequency Domain (Raw)")
        axs[3].set_xlim([0, 4000])
        axs[3].set_xlabel("Frequency (Hz)")
        axs[3].set_ylabel("Magnitude")

        # --- Plot 5: Frequency Domain (Cleaned) ---
        axs[4].cla()
        fft_clean = np.abs(np.fft.rfft(clean_copy))
        freqs_clean = np.fft.rfftfreq(len(clean_copy), d=1/SAMPLE_RATE)
        axs[4].plot(freqs_clean, fft_clean)
        axs[4].set_title("Frequency Domain (Cleaned)")
        axs[4].set_xlim([0, 4000])
        axs[4].set_xlabel("Frequency (Hz)")
        axs[4].set_ylabel("Magnitude")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.05)

# === Feature extraction ===
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


# === Audio thread ===
def start_audio_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        sd.sleep(1000000)

# === Main ===
try:
    threading.Thread(target=start_audio_stream, daemon=True).start()
    plot_live()
except KeyboardInterrupt:
    print("\nInterrupted by user.")
