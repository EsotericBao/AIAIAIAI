import os
import numpy as np
import pyaudio
import whisper
import torch
import threading
import time
import noisereduce as nr
import sys
from tkinter import messagebox


class ToggleEars:
    def __init__(self):
        # Load Whisper model
        self.model = whisper.load_model("turbo")  # Use "medium" for better accuracy

        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper's expected sample rate
        self.CHUNK = 1024  # Buffer size

        # Audio storage
        self.audio_buffer = []  # Store entire recording session
        self.running = False  # Initially not recording

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

    
    def start(self):
        """Start recording audio continuously."""
        if not self.running:
            self.running = True
            self.audio_buffer = []  # Clear previous recordings

            # Start audio stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )

            # Start recording thread
            self.audio_thread = threading.Thread(target=self.record_audio_stream, daemon=True)
            self.audio_thread.start()
            print("🎤 Recording started...")
            return "🎤 Recording started..."

    def record_audio_stream(self):
        """Continuously records audio until 'Stop' is clicked."""
        while self.running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(data)  # Store raw PCM audio
            except IOError:
                print("⚠️ Error: Audio buffer overflow")

    def stop(self):
        """Stop recording and transcribe the full audio."""
        if self.running:
            self.running = False
            print("\n🛑 Stopping recording...")
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"⚠️ Error closing audio stream: {e}")

            print("✅ Recording stopped. Transcribing...")

            # Process the full recording
            return self.transcribe_audio()

    def transcribe_audio(self):
        """Processes the recorded audio and transcribes it using Whisper."""
        if not self.audio_buffer:
            print("⚠️ No audio recorded.")
            return

        # Convert recorded PCM data to NumPy array
        raw_audio = np.frombuffer(b''.join(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

        # Apply noise reduction
        #processed_audio = nr.reduce_noise(y=raw_audio, sr=self.RATE)
        prompt = "This is a question about the Singapore Police Force."
        # Transcribe using Whisper
        result = self.model.transcribe(raw_audio, fp16=torch.cuda.is_available(),  language="en", initial_prompt=prompt)
        text = result["text"].strip()

        print(f"📝 Transcription: {text}")
        return text


if __name__ == "__main__":
    pass