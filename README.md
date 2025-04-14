# AI Chatbot Project

## Project Overview
This project builds an interactive AI chatbot named **CHATBOT**, which leverages:
- **LangChain** for managing retrieval-based question answering.
- **ChromaDB** for storing and querying document embeddings.
- **Ollama LLM** for natural language understanding and response generation.
- **OpenAI Whisper** for voice input transcription.
- **Google Cloud Text-to-Speech (TTS)** for voice output.
- **Optimized PaddleOCR** for accurate text extraction from scanned documents.
- **Multiprocessing and Dynamic Chunking** for efficient PDF and text file ingestion.
- **Streamlit UI** for an interactive user interface.


CHATBOT is designed to assist users by answering questions based strictly on a given context, with professional tone and concise responses.

---

## Features
- **Context-Based QA**: Answers questions using context retrieved from embedded documents.
- **Voice Interaction**:
  - Transcribe user speech via OpenAI Whisper.
  - Generate audio responses via Google Cloud TTS.
- **Customizable Personality**: Configurable chatbot prompt and response style.
- **Document Integration**:
  - PDF and **text file ingestion** and embedding into ChromaDB for retrieval.
  - **Optimized OCR with PaddleOCR** for enhanced accuracy in scanned documents.
  - **Dynamic chunking** of documents for improved query matching and retrieval performance.
  - **Multiprocessing support** for faster processing of large documents.
- **Interactive UI**:
  - Streamlit-based front-end for user-friendly interaction.
  - Real-time speech-to-text and chatbot response streaming.

---

## Prerequisites

### Tools and Libraries
1. **Python (3.12)**
2. **Set up a Virtual Environment**:
   ```bash
   python -m pip install virtualenv
   virtualenv --python="3.12"
   source venv/bin/activate
   ```
3. **Install Required Libraries**:
    - install [ffmpeg](https://ffmpeg.org/) and add to PATH
    - install [pytorch](https://pytorch.org/)
   ```bash
   pip install -r requirements.txt
   ```
   Ignore dependency conflict for protobuf
   ```bash 
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. streamlit-extras 0.6.0 requires protobuf!=3.20.2, but you have protobuf 3.20.2 which is incompatible. 
   ```
   

### Google Cloud Setup
1. **Enable Text-to-Speech API**:
   - Navigate to [Google Cloud Console](https://console.cloud.google.com/).
   - Enable the **Text-to-Speech API**.
2. **Service Account Key**:
   - Download a JSON key for a service account with the `Text-to-Speech API User` role.
   - Set the environment variable in config.py:
     ```bash
     GOOGLE_CLOUD_STT_KEY = os.path.join(BASE_DIR, "keys\your-service-account-key.json")
     ```

### Ollama Setup
1. **Install Ollama**:
   - Download and install Ollama from [Ollama's website](https://ollama.ai/).
   - Ensure `ollama` is added to your system's PATH.
2. **Serve a Model**:
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Ensure the server is running in the background.
3. **Run a Model**:
   - Pull the desired model (e.g., `llama3` or `custom-model`):
     ```bash
     ollama pull llama3
     ```
   - Verify the model is ready for use by querying:
     ```bash
     ollama chat llama3
     ```

---

## Project Components

### **1. Chatbot Script (`chatbot.py`)**
Main features:
- **Prompt Template**:
  - Defines the personality and tone of CHATBOT.
- **LangChain RetrievalQA**:
  - Uses ChromaDB as a vector database for document retrieval.
- **Dynamic Session Handling**:
  - Allows resetting the chat history.

#### Key Configurations:
- **ChromaDB**: Stores and retrieves document embeddings.
- **Ollama LLM**: Generates responses based on user input and retrieved context.
- **Prompt Template**:
  ```python
  """
  Your name is CHATBOT, a helpful assistant.
  Always maintain professionalism and a concise tone in your responses.

  Guidelines:
  - Base your answers solely on the retrieved context.
  - If the context lacks enough information, respond: "I couldn't find relevant information to answer your question."
  - Do not fabricate information.
  - Keep responses short and professional.
  """
  ```

### **2. Document Ingestion Script (`document_ingest.py`)**
Handles document ingestion and embedding into ChromaDB.
- **Processes both PDFs and text files.**
- **Optimized OCR with PaddleOCR**:
  - **Image preprocessing** (adaptive thresholding, sharpening, denoising).

  - **Multiprocessing for faster OCR processing.**
- **Duplicate Prevention**: Uses file hashes to skip re-ingestion of already-processed documents.
- **Dynamic and Recursive Chunking**:
  - Texts are dynamically chunked based on their length to optimize retrieval.
  - Recursively splits large chunks into smaller sub-chunks to ensure context integrity.

### **3. Config File (`config.py`)**
Defines standard directory paths for seamless integration:
- `PDF_FOLDER`: Path to the folder containing PDFs.
- `TEXT_FOLDER`: Directory for storing extracted text files.
- `CHROMADB`: Directory for ChromaDB persistence.
- `MAINDB`, `FALLBACKDB`: Separate databases for different document types.

---

## Usage Instructions

### 1. **Ingest PDFs**
- Place PDFs and text files in the directory specified by `PDF_FOLDER` or `TEXT_FOLDER`.
- Run the ingestion script:
  ```bash
  py document_ingest.py
  ```
Change use_gpu to False if not using gpu
```bash
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
```
- Embedded data is stored in `CHROMADB`.

### 2. **Start the Streamlit UI**
Run the following command:
```bash
streamlit run app.py
```
This will launch the chatbot interface in a browser.

### 3. **Run the Chatbot from CLI (Optional)**
Start the chatbot interaction:
```bash
py chatbot.py
```
- Ask CHATBOT questions based on the provided context.
- Use commands:
  - `reset` or `clear` to reset the chat history.
  - `exit` to end the session.


---

## Customization

### Adjust Voice and Tone
Modify the `prompt` in `chatbot.py` to change CHATBOT's personality.

### Change Voice Settings
- Adjust **speed** and **pitch** in Google Cloud TTS:
  ```python
  audio_config = texttospeech.AudioConfig(
      speaking_rate=1.2,  # Adjust speed (default 0.88)
      pitch=2.0,          # Adjust pitch (default 2.0)
  )
  ```

---

## Troubleshooting

### Common Issues
1. **Environment Variable Not Set**:
   - Ensure `GOOGLE_APPLICATIONS_CREDENTIALS` is set correctly.

2. **API Errors**:
   - Confirm the **Text-to-Speech API** is enabled for your Google Cloud project.

3. **Audio Playback Issues**:
   - Install `ffmpeg` for playing generated MP3 files:
     ```bash
     py -m pip install ffmpeg-python
     ```

4. **No Results from ChromaDB**:
   - Ensure documents are correctly embedded into ChromaDB during ingestion.

---

## Project Files
1. **`chatbot.py`**: Main chatbot script.
2. **`document_ingest.py`**: Document ingestion and embedding script.
3. **`config.py`**: Configuration file for directory paths.
4. **`app.py`**: Streamlit UI for chatbot interaction.
5. **Google Cloud JSON Key**: Service account key for API access.

---
