# AI Chatbot Project

## Project Overview
This project builds an interactive AI chatbot named **CHATBOT**, which leverages:
- **LangChain** for managing retrieval-based question answering.
- **ChromaDB** for storing and querying document embeddings.
- **Ollama LLM** for natural language understanding and response generation.
- **OpenAI Whisper** for voice input transcription.
- **Google Cloud Text-to-Speech (TTS)** for voice output.


CHATBOT is designed to assist users by answering questions based strictly on a given context, with professional tone and concise responses.

---

## Features
- **Context-Based QA**: Answers questions using context retrieved from embedded documents.
- **Voice Interaction**:
  - Transcribe user speech via OpenAI Whisper.
  - Generate audio responses via Google Cloud TTS.
- **Customizable Personality**: Configurable chatbot prompt and response style.
- **Document Integration**:
  - PDF ingestion and embedding into ChromaDB for retrieval.
  - Dynamic chunking of documents for improved query matching.

---

## Prerequisites

### Tools and Libraries
1. **Python (3.12)**
2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Microsoft Visual Studio Setup
1. Download and install **Microsoft Visual Studio** from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/).
2. During installation, ensure the following workloads are selected:
   - **.NET desktop development**
   - **Desktop development with C++**
3. Once installed, restart your terminal or IDE to ensure the environment variables are updated.


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
     ```bash
     ollama pull mxbai-embed-large
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

### **2. PDF Ingestion Script (`gpu_ingest.py`)**
Handles document ingestion and embedding into ChromaDB.
- **Dynamic Language Handling**: Detects document languages (English, Chinese, Malay, Tamil) and processes accordingly.
- **OCR Support**: Extracts text from scanned PDFs using PaddleOCR.
- **Duplicate Prevention**: Uses file hashes to skip re-ingestion of already-processed documents.

### **3. Config File (`config.py`)**
Defines standard directory paths for seamless integration:
- `PDF_FOLDER`: Path to the folder containing PDFs.
- `CHROMADB`: Directory for ChromaDB persistence.
- `TEXTS_FOLDER`: Directory for storing extracted text files.

---

## Usage Instructions

### 1. **Ingest PDFs**
- Place PDFs in the directory specified by `PDF_FOLDER`.

- Run the `gpu_ingest.py` script to process and embed documents (`cpu_ingest.py` if not using gpu):
```bash
py gpu_ingest.py
```
- Embedded data is stored in `CHROMADB`.

### 2. **Run the Chatbot**
Start the chatbot interaction:
```bash
py chatbot.py
```
- Ask CHATBOT questions based on the provided context.
- Use commands:
  - `reset` or `clear` to reset the chat history.
  - `exit` to end the session.

### 3. **Voice Interaction (Optional)**
- Whisper transcribes microphone input.
- Google Cloud TTS generates audio responses.

---

## Customization

### Adjust Voice and Tone
Modify the `prompt` in `chatbot.py` to change CHATBOT's personality.

### Change Voice Settings
- Adjust **speed** and **pitch** in Google Cloud TTS:
  ```python
  audio_config = texttospeech.AudioConfig(
      speaking_rate=1.2,  # Adjust speed (default 1.0)
      pitch=2.0,          # Adjust pitch (default 1.0)
  )
  ```

### Chunk Size for PDFs
Update chunk size and overlap in `pdf_ingest.py`:
```python
chunk_size = 1000
chunk_overlap = 50
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
2. **`gpu_ingest.py`**: PDF ingestion and embedding script.
3. **`config.py`**: Configuration file for directory paths.
4. **Google Cloud JSON Key**: Service account key for API access.

---

