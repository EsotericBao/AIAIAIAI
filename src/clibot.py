import os
import librosa
import numpy as np
import pyaudio
import whisper
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
from scipy.signal import wiener
from google.cloud import texttospeech
from config import GOOGLE_CLOUD_STT_KEY, RECORDINGS_FOLDER
from chatbot import Chatbot
from ariel import TextToSpeech
import time

class VoiceBox:
    def __init__(self): 
        os.makedirs(RECORDINGS_FOLDER, exist_ok=True)  # Create the folder if it doesn’t exist
        # Initialise the chatbot
        self.chatbot = Chatbot()
        self.tts = TextToSpeech()
        # Load Whisper model
        self.model = whisper.load_model("turbo")
        # Configure audio output
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.1,  # Slightly slower for clarity
            pitch=1.5,
        )
        #Initialise audio variables
        
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper's expected sample rate
        self.CHUNK = 16000 #1024
        #self.audio = pyaudio.PyAudio()

    # Function to synthesise speech using Google Cloud TTS
    def synthesise_response(self, text, output_file):
        """
        Convert chatbot response to speech using Google Cloud TTS.
        """
        try:
            self.client = texttospeech.TextToSpeechClient()
            
            # Set up input text
            input_text = texttospeech.SynthesisInput(text=text)
            print("Generating speech...")
            response = self.client.synthesize_speech(input=input_text, voice=self.voice, audio_config=self.audio_config)

            # Save the audio
            with open(output_file, "wb") as out:
                out.write(response.audio_content)
            print(f"Audio saved to {output_file}")

            # Play the audio
            os.system(f"ffplay -nodisp -autoexit {output_file}")
            
        except Exception as e:
            print(f"Error during TTS synthesis: {e}")

    def normalise_audio(self, audio_data, target_dBFS=-20.0):
        """
        normalise the audio to a target decibel level (dBFS).
        
        Args:
            audio_data (numpy.ndarray): The raw audio data.
            target_dBFS (float): Target loudness level in dBFS.
        
        Returns:
            numpy.ndarray: The normalised audio data.
        """
        rms = np.sqrt(np.mean(audio_data**2))  # Compute RMS level
        scalar = 10**(target_dBFS / 20) / rms  # Compute scaling factor
        normalised_audio = audio_data * scalar
        return np.clip(normalised_audio, -1.0, 1.0)  # Clip to prevent distortion

    # Function to record audio with silence detection
    def record_until_silence(self, output_filename, silence_threshold, silence_duration):
        """
        Record audio using PyAudio and save when silence is detected.
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)

        print("🎤 Recording... Speak now!")
        frames = []
        silent_chunks = 0
        is_speaking = False

        while True:
            try:
                data = np.frombuffer(stream.read(self.CHUNK, exception_on_overflow=False), dtype=np.int16).astype(np.float32) / 32768.0
            except IOError:
                print("Error: Audio buffer overflow")
                continue
            frames.append(data)

            # Compute root mean square (RMS) for better silence detection
            rms = np.sqrt(np.mean(data**2))

            # Check if below silence threshold
            if rms < silence_threshold:
                silent_chunks += 1
                if is_speaking and silent_chunks >= (self.RATE / self.CHUNK * silence_duration):
                    break
            else:
                silent_chunks = 0
                is_speaking = True

        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Concatenate all frames
        audio_data = np.concatenate(frames)

        # Apply noise reduction
        reduced_audio = nr.reduce_noise(y=audio_data, sr=self.RATE)

        # Apply volume normalisation
        normalised_audio = self.normalise_audio(reduced_audio)

        # Save the processed audio
        sf.write(output_filename, normalised_audio, self.RATE)

        # Convert to WAV using pydub for compatibility
        sound = AudioSegment.from_wav(output_filename)
        sound.export(output_filename, format="wav", parameters=["-ar", "16000", "-ac", "1"])

        print(f"Audio saved to {output_filename}")



    # Function to preprocess audio with noise reduction
    def preprocess_audio(self, input_filename, output_filename):
        """
        Load audio, apply noise reduction, normalise, and resample to 16kHz.
        """
        print("Preprocessing audio with librosa...")
        audio, sr = librosa.load(input_filename, sr=16000)
        audio = librosa.util.normalize(audio)  # normalise volume
        audio = wiener(audio)  # Noise reduction

        # Save processed audio
        sf.write(output_filename, audio, sr)
        print("Audio preprocessing complete.")

    # Function to transcribe audio using Whisper
    def transcribe_audio(self,filename):
        """
        Transcribe preprocessed audio using Whisper.
        """
        print("🔄 Transcribing audio...")
        prompt = "This is a question about the Singapore Police Force."
        result = self.model.transcribe(filename, language="en", initial_prompt=prompt)
        transcription = result["text"].strip()
        print(f"Transcription: {transcription}")
        return transcription
    
    def remove_think(self, response):
        delimiter = "</think>"
        answer = response.split(delimiter)
        return answer[len(answer)-1].strip("\n"
                                           )
        # Main loop to integrate everything
    def chat(self):
        raw_file = os.path.join(RECORDINGS_FOLDER, "raw_input.wav")
        processed_file = os.path.join(RECORDINGS_FOLDER, "processed_input.wav")
        output_file = os.path.join(RECORDINGS_FOLDER, "response.wav")
        try:
            trigger = input("Trigger: ")
            if trigger:
                '''
                silence_treshold
                0.01 → Sensitive (Detects even soft whispers)
                0.1 → Normal (Good for most environments)
                0.3 → Strict (Ignores soft background noise)
                '''
                self.record_until_silence(output_filename=raw_file, silence_threshold=0.3, silence_duration=0.5   )

                # Preprocess the audio
                #self.preprocess_audio(raw_file, processed_file)

                # Transcribe the processed audio
                user_query = self.transcribe_audio(raw_file)
                #user_query = trigger
                # Exit if user says "quit"
                if trigger or user_query.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                else:
                    print(f"USER: {user_query}")
                    # Get response from chatbot
                    response = self.chatbot.get_response(user_query)
                    #print(response, end='\n')
                    answer = self.remove_think(response)
                    print(f"CHAT: {answer}")
                    # synthesise and play response
                    #self.tts.synthesize_speech(answer, file=output_file)
        except KeyboardInterrupt:
            return("Ending")

    def main(self,):
        

        print("Welcome to the Voice-Enabled Chatbot!")
        print("Speak to ask your question. Say 'quit' to exit.")

        while True:
            self.chat()


if __name__ == "__main__":
    ariel = VoiceBox()
    ariel.main()