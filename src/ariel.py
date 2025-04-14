import os
from google.cloud import texttospeech
from config import GOOGLE_CLOUD_STT_KEY, RECORDINGS_FOLDER

class TextToSpeech:
    def __init__(self, language_code="en-GB", voice_name="en-GB-Neural2-F", speaking_rate=1.1, pitch=1.5):
        """
        Initializes the Text-to-Speech client with default settings.

        Args:
            language_code (str): Language code for the voice.
            voice_name (str): Specific Google Cloud voice name.
            speaking_rate (float): Speed of speech (default 1.0).
            pitch (float): Pitch adjustment (default 0.0).
        """
        # Set Google Cloud credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_STT_KEY
        
        # Initialize client
        self.client = texttospeech.TextToSpeechClient()

        # Configure voice parameters
        self.voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )

        # Configure audio settings
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
        )

    def synthesise_speech(self, text, file="voice.wav"):
        """
        Converts text to speech and saves it as a wav file. 

        Args:
            text (str): The text to convert to speech.
            output_file (str): The filename for the output audio.
        """
        output_file = os.path.join("recordings", file)
        print(output_file)
        try:
            # Set up input text
            input_text = texttospeech.SynthesisInput(text=text)

            # Synthesize speech
            response = self.client.synthesize_speech(
                input=input_text, voice=self.voice_params, audio_config=self.audio_config
            )

            # Save audio to a file
            with open(output_file, "wb") as out:
                out.write(response.audio_content)
            print(f"✅ Audio content saved to {output_file}")
            # Play the audio
            os.system(f"ffplay -nodisp -autoexit {output_file}")
        except Exception as e:
            print(f"❌ Error: {e}")

# Test the class when run as a standalone script
if __name__ == "__main__":
    tts = TextToSpeech()
    sample_text = "Hello world"
    tts.synthesise_speech(sample_text)
