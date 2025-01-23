import os
from google.cloud import texttospeech
from config import GOOGLE_CLOUD_STT_KEY

# Set the path to your Google Cloud JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_STT_KEY

def synthesize_speech_with_journey_voice(text, output_file="journey_voice.mp3"):
    """
    Synthesize speech using the Journey voice in Google Cloud Text-to-Speech.

    Args:
        text (str): Text to convert to speech.
        output_file (str): Path to save the audio file.
    """
    try:
        client = texttospeech.TextToSpeechClient()

        # Set up input text
        input_text = texttospeech.SynthesisInput(text=text)

        # Configure the Journey voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-TW",  # Adjust for the desired language if needed #cmn-TW #en-GB
            name="cmn-TW-Wavenet-A",  # 'J' corresponds to the Journey voice #cmn-TW-Wavenet-A #en-GB-Neural2-F
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,  # FEMALE voice
        )

        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.88,  # Adjust speed (default is 1.0)
            pitch=2,          # Adjust pitch (default is 0.0)
        )

        # Synthesize speech
        print("Generating speech using the Journey voice...")
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        # Save the audio to a file
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Prompt the user for text input
    print("Enter the text you want to synthesize into speech with the Journey voice:")
    user_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
    # Synthesize the speech using the Journey voice
    synthesize_speech_with_journey_voice(t2, "journey_voice.mp3")

    # Notify user about the output file
    print("\nThe synthesized speech has been saved as 'journey_voice.mp3'.")
    print("You can play it using your preferred media player.")
