import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import atexit
from scipy.signal import spectrogram
import sounddevice as sd
import wave
import torch
from df import init_df, enhance
import joblib
import librosa
from scipy.signal import butter, lfilter

class AudioEmotionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Audio Emotion Detection")
        self.root.geometry("1200x900")
        
        # === Audio Processing Parameters ===
        self.SAMPLE_RATE = 16000
        self.BLOCKSIZE = 16000
        self.DENOISE_BLEND = 6.0
        self.SILENCE_THRESHOLD = 0.015
        
        # === Data Storage ===
        self.audio_buffer = np.zeros(self.BLOCKSIZE)
        self.latest_cleaned_audio = np.zeros(self.BLOCKSIZE)
        self.raw_audio_storage = []
        self.clean_audio_storage = []
        self.buffer_lock = threading.Lock()
        
        # === Status Variables ===
        self.is_running = False
        self.current_emotion = tk.StringVar(value="No emotion detected")
        self.current_confidence = tk.StringVar(value="0.00")
        self.denoise_latency = tk.StringVar(value="0.00")
        self.emotion_latency = tk.StringVar(value="0.00")
        
        # === Load Models ===
        self.load_models()
        
        # === Setup GUI ===
        self.setup_gui()
        
        # === Setup Exit Handler ===
        atexit.register(self.on_exit)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_models(self):
        """Load emotion detection and denoising models"""
        try:
            self.mlp_model = joblib.load("mlp_emotion_classifier_best_model2.joblib")
            self.encoder = joblib.load("emotion_encoder.joblib")
            self.scaler = joblib.load("feature_scaler.joblib")
            
            # Initialize DeepFilterNet
            self.model, self.df_state, _ = init_df()
            
            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load models: {str(e)}")
            self.root.destroy()
            
    def setup_gui(self):
        """Setup the main GUI layout"""
        # === Main Frame ===
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === Control Panel ===
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(control_frame, text="Save Audio", command=self.save_audio_files)
        self.save_button.grid(row=0, column=2)
        
        # === Status Panel ===
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Emotion Display
        ttk.Label(status_frame, text="Current Emotion:").grid(row=0, column=0, sticky=tk.W)
        emotion_label = ttk.Label(status_frame, textvariable=self.current_emotion, font=("Arial", 12, "bold"))
        emotion_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(status_frame, text="Confidence:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        confidence_label = ttk.Label(status_frame, textvariable=self.current_confidence, font=("Arial", 12))
        confidence_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        
        # Latency Display
        ttk.Label(status_frame, text="Denoise Latency:").grid(row=1, column=0, sticky=tk.W)
        denoise_label = ttk.Label(status_frame, textvariable=self.denoise_latency, font=("Arial", 10))
        denoise_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(status_frame, text="Emotion Latency:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0))
        emotion_lat_label = ttk.Label(status_frame, textvariable=self.emotion_latency, font=("Arial", 10))
        emotion_lat_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))
        
        # === Plots Frame ===
        plots_frame = ttk.Frame(main_frame)
        plots_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 10), dpi=80)
        self.fig.suptitle("Real-Time Audio Visualization", fontsize=14)
        
        # Create subplots
        self.axs = []
        for i in range(5):
            ax = self.fig.add_subplot(5, 1, i+1)
            self.axs.append(ax)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def bandpass_filter(self, data, low=300, high=3400, fs=None):
        """Apply bandpass filter to audio data"""
        if fs is None:
            fs = self.SAMPLE_RATE
        nyq = 0.5 * fs
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
        return lfilter(b, a, data)
    
    def normalize(self, audio):
        """Normalize audio to [-1, 1] range"""
        return audio / (np.max(np.abs(audio)) + 1e-6)
    
    def preprocess(self, audio):
        """Preprocess audio before denoising"""
        filtered = self.bandpass_filter(audio)
        normalized = self.normalize(filtered)
        return normalized
    
    def denoise(self, audio):
        """Denoise audio using DeepFilterNet"""
        tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        if torch.cuda.is_available():
            tensor_audio = tensor_audio.to("cuda")
            self.model.to("cuda")
        with torch.no_grad():
            output = enhance(self.model, self.df_state, tensor_audio.cpu())
        return output.numpy().flatten()
    
    def is_silence(self, audio, threshold=None):
        """Check if audio segment is silence"""
        if threshold is None:
            threshold = self.SILENCE_THRESHOLD
        energy = np.sqrt(np.mean(audio ** 2))
        return energy < threshold
    
    def extract_features_from_array(self, y, sr=16000):
        """Extract audio features for emotion detection"""
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
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback function"""
        # Store raw audio immediately
        audio_data = indata[:, 0].copy()
        
        with self.buffer_lock:
            self.audio_buffer = audio_data
            self.raw_audio_storage.append(audio_data)
        
        # Process audio in background thread to avoid blocking
        if not hasattr(self, 'processing_thread') or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_audio, args=(audio_data,), daemon=True)
            self.processing_thread.start()
    
    def process_audio(self, audio_data):
        """Process audio in background thread"""
        try:
            # Preprocessing
            processed = self.preprocess(audio_data)
            
            # Denoising with timing
            start_denoise = time.perf_counter()
            cleaned = self.denoise(processed)
            denoise_time = (time.perf_counter() - start_denoise) * 1000
            
            # Conditional blending based on silence
            if self.is_silence(processed):
                output_audio = np.zeros_like(cleaned)
            else:
                output_audio = self.DENOISE_BLEND * cleaned + (1 - self.DENOISE_BLEND) * processed
            
            with self.buffer_lock:
                self.latest_cleaned_audio = output_audio
                self.clean_audio_storage.append(output_audio.copy())
            
            # Emotion prediction with timing
            start_emotion = time.perf_counter()
            features = self.extract_features_from_array(output_audio, sr=self.SAMPLE_RATE)
            features_scaled = self.scaler.transform([features])
            probs = self.mlp_model.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]
            emotion = self.encoder.inverse_transform([predicted_idx])[0]
            emotion_time = (time.perf_counter() - start_emotion) * 1000
            
            # Schedule GUI update on main thread
            self.root.after(0, self.update_emotion_display, emotion, confidence, denoise_time, emotion_time)
            
        except Exception as e:
            print(f"[Audio Processing Error] {e}")
    
    def update_emotion_display(self, emotion, confidence, denoise_time, emotion_time):
        """Update emotion display on main thread"""
        self.current_emotion.set(emotion)
        self.current_confidence.set(f"{confidence:.2f}")
        self.denoise_latency.set(f"{denoise_time:.2f} ms")
        self.emotion_latency.set(f"{emotion_time:.2f} ms")
    
    def update_plots(self):
        """Update all visualization plots"""
        if not self.is_running:
            return
        
        try:
            with self.buffer_lock:
                buffer_copy = self.audio_buffer.copy()
                clean_copy = self.latest_cleaned_audio.copy()
            
            # Clear all subplots
            for ax in self.axs:
                ax.clear()
            
            # Plot 1: Time Domain (Raw)
            self.axs[0].plot(buffer_copy)
            self.axs[0].set_title("Time Domain (Raw)")
            self.axs[0].set_ylim([-1, 1])
            self.axs[0].set_ylabel("Amplitude")
            
            # Plot 2: Spectrogram (Raw)
            f_raw, t_raw, Sxx_raw = spectrogram(buffer_copy, self.SAMPLE_RATE, nperseg=256, noverlap=128)
            im1 = self.axs[1].imshow(10 * np.log10(Sxx_raw + 1e-10), origin='lower', aspect='auto',
                                     extent=[t_raw[0], t_raw[-1], f_raw[0], f_raw[-1]], cmap='viridis')
            self.axs[1].set_title("Spectrogram (Raw)")
            self.axs[1].set_ylabel("Frequency (Hz)")
            self.axs[1].set_ylim([0, 4000])
            
            # Plot 3: Spectrogram (Cleaned)
            f_clean, t_clean, Sxx_clean = spectrogram(clean_copy, self.SAMPLE_RATE, nperseg=256, noverlap=128)
            im2 = self.axs[2].imshow(10 * np.log10(Sxx_clean + 1e-10), origin='lower', aspect='auto',
                                    extent=[t_clean[0], t_clean[-1], f_clean[0], f_clean[-1]], cmap='viridis')
            self.axs[2].set_title("Spectrogram (Cleaned)")
            self.axs[2].set_ylabel("Frequency (Hz)")
            self.axs[2].set_ylim([0, 4000])
            
            # Plot 4: Frequency Domain (Raw)
            fft_raw = np.abs(np.fft.rfft(buffer_copy))
            freqs_raw = np.fft.rfftfreq(len(buffer_copy), d=1/self.SAMPLE_RATE)
            self.axs[3].plot(freqs_raw, fft_raw)
            self.axs[3].set_title("Frequency Domain (Raw)")
            self.axs[3].set_xlim([0, 4000])
            self.axs[3].set_ylabel("Magnitude")
            
            # Plot 5: Frequency Domain (Cleaned)
            fft_clean = np.abs(np.fft.rfft(clean_copy))
            freqs_clean = np.fft.rfftfreq(len(clean_copy), d=1/self.SAMPLE_RATE)
            self.axs[4].plot(freqs_clean, fft_clean)
            self.axs[4].set_title("Frequency Domain (Cleaned)")
            self.axs[4].set_xlim([0, 4000])
            self.axs[4].set_xlabel("Frequency (Hz)")
            self.axs[4].set_ylabel("Magnitude")
            
            # Adjust layout and redraw
            self.fig.tight_layout(rect=[0, 0, 1, 0.96])
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")
        
        # Schedule next update
        if self.is_running:
            self.root.after(100, self.update_plots)  # Increased interval to 100ms
    
    def start_detection(self):
        """Start audio detection and processing"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCKSIZE
            )
            self.stream.start()
            
            # Start plot updates
            self.update_plots()
            
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to start audio stream: {str(e)}")
            self.stop_detection()
    
    def stop_detection(self):
        """Stop audio detection and processing"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        # Stop audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        # Reset status
        self.current_emotion.set("No emotion detected")
        self.current_confidence.set("0.00")
        self.denoise_latency.set("0.00")
        self.emotion_latency.set("0.00")
    
    def save_audio_files(self):
        """Save recorded audio to WAV files"""
        try:
            self.save_audio("radio_raw.wav", self.raw_audio_storage)
            self.save_audio("radio_cleaned.wav", self.clean_audio_storage)
            messagebox.showinfo("Save Complete", "Audio files saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save audio files: {str(e)}")
    
    def save_audio(self, filename, data):
        """Save audio data to WAV file"""
        if not data:
            return
            
        flat = np.concatenate(data)
        scaled = np.int16(flat / (np.max(np.abs(flat)) + 1e-6) * 32767)
        
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(scaled.tobytes())
    
    def on_exit(self):
        """Handle application exit"""
        print("\nSaving audio...")
        if self.raw_audio_storage and self.clean_audio_storage:
            self.save_audio("radio_raw.wav", self.raw_audio_storage)
            self.save_audio("radio_cleaned.wav", self.clean_audio_storage)
        print("Done.")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.on_exit()
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = AudioEmotionGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")