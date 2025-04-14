import os
import numpy as np
import time
import torch
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.streaming_write import write

from json import load
import base64

from toggleears import ToggleEars  # Import recording
from chatbot import Chatbot  # Import chatbot
from ariel import TextToSpeech #import speech

class APP:
    def __init__(self): 
        startup = False
        if not startup:
            print("App starting")
            startup = True

    def load_css(self, file_name):
        """Inject custom CSS for styling."""
        with open(file_name, "r") as f:
            return f"<style>{f.read()}</style>"

    def get_base64_image(self, image_path):
        """Encode image to Base64 for embedding in Streamlit UI."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    def load_lottie_file(self, filepath):
        """Load Lottie animation from a local JSON file."""
        with open(filepath, "r") as f:
            return load(f)

    def app_innit(self):
        """Initialize the Streamlit UI and chatbot."""
        # Store instances in session state (to avoid reinitialization)
        if "ears" not in st.session_state:
            st.session_state.ears = ToggleEars()  # Persistent SST instance
        if "llm" not in st.session_state:
            st.session_state.llm = Chatbot()  # Persistent chatbot instance
        if "mouth" not in st.session_state:
            st.session_state.mouth = TextToSpeech()  # Persistent TTS instance
        if "status" not in st.session_state:
            st.session_state.status = "Idle"  # Default status
        if "user_query" not in st.session_state:
            st.session_state.user_query = "Transcribed audio..."
        if "response" not in st.session_state:
            st.session_state.response = ""
        if "recording_started" not in st.session_state:
            st.session_state.recording_started = False  # Locks Stop & Transcribe initially
        if "transcription_done" not in st.session_state:
            st.session_state.transcription_done = False  # Locks Process Query initially

        # Load external CSS
        image_path = "assets/logo.png"
        image_base64 = self.get_base64_image(image_path)
        css = self.load_css("styles/styles.css")

        # Display UI elements
        st.markdown("<h2 class='title'>🎙️ FOURIER BOT 🗣️</h2>", unsafe_allow_html=True)
        st.markdown(css, unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/png;base64,{image_base64}" class="breathing-logo">', unsafe_allow_html=True)

        status_color = "status-idle"
        if "Recording" in st.session_state.status:
            status_color = "status-recording"
        elif "Processing" in st.session_state.status:
            status_color = "status-processing"

        st.markdown(f'<div class="status-bar {status_color}">{st.session_state.status}</div>', unsafe_allow_html=True)

        self.chatbot_ui()
        print("App started")

    def chatbot_ui(self):
        """Define chatbot UI with recording and transcription buttons."""
        def start():
            """Start recording."""
            status = st.session_state.ears.start()
            st.session_state.status = status
            st.session_state.recording_started = True

        def stop():
            """Stop recording and transcribe."""
            st.session_state.status = "🔄 Processing Query..."
            with st.spinner("⏳ Transcribing...", show_time=True):
                query = st.session_state.ears.stop()
                st.session_state.user_query = query
                st.session_state.status = "⏺️Idle⏺️"
                st.session_state.transcription_done = True  # Unlock Process Query
                st.session_state.recording_started = False

        def process_query():
            """Send query to chatbot and display response."""
            print("Sending query")
            with st.spinner("🤖 Chatbot is thinking...", show_time=True):
                st.session_state.response = st.session_state.llm.get_response(st.session_state.user_query)
                st.session_state.recording_started = False
                st.session_state.transcription_done = False
            with st.spinner("Speaking"):
                st.session_state.mouth.synthesise_speech(st.session_state.response)
            

        # **Button Locking Logic**
        disable_start = st.session_state.recording_started  # Disable Start when recording has started
        disable_stop = not st.session_state.recording_started # Enable Stop only if recording has started
        disable_process_query = not st.session_state.transcription_done # Enable Process Query only if transcription is done

        a, b, c = st.columns([1, 1, 1])
        with a:
            st.button("🟢 Start Recording ✅", on_click=start, key="start_btn", help="Start recording your voice", disabled=disable_start)
        with b:     
            st.button("🔴 Stop & Transcribe", on_click=stop,  key="stop_btn", help="Stop and transcribe audio", disabled=disable_stop)
        with c:
            st.button("🔄 Process Query", on_click=process_query, key="process_btn", help="Send transcription to AI", disabled=disable_process_query)

        #st.markdown("___")

        # Display transcribed query
        st.markdown("<h3 class='subsection'>📝 Transcription</h3>", unsafe_allow_html=True)
        st.write(f"<div style='text-align:center; font-size:15px;'><b>{st.session_state.user_query}</b></div>", unsafe_allow_html=True)


        # Display chatbot response
        st.markdown("<h3 class='subsection'>💬 Chatbot Response</h3>", unsafe_allow_html=True)
        st.write(f"<div style='text-align:center; font-size:15px;'><b>{st.session_state.response}</b></div>", unsafe_allow_html=True)
 

    def main(self):
        """Run the Streamlit UI."""
        self.app_innit()

if __name__ == "__main__":
    try:
        app = APP()
        app.main()
    except Exception as e:
        print(e)