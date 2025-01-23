import os

# Define base directories
SRC_DIR = os.path.abspath(os.path.dirname(__file__))  # Directory of the config file
BASE_DIR = os.path.dirname(SRC_DIR)
#PDF folders
PDF_FOLDER = os.path.join(SRC_DIR, "PDFs")
MULTI_PDF = os.path.join(PDF_FOLDER, "Multilanguage")

#Text folders
TEXTS_FOLDER = os.path.join(SRC_DIR, "output_texts")

#Vector Database folders
VECTORDB = os.path.join(SRC_DIR, "vector_database")
CHROMADB = os.path.join(VECTORDB, "chromadb")
TESTDB = os.path.join(SRC_DIR, "testdb")
COLLECTION ="pdf_embeddings"

#Keys
GOOGLE_CLOUD_STT_KEY = os.path.join(SRC_DIR, "keys/key.json")

#
#
#
#Set up llm models and embedding models
from langchain_ollama import OllamaEmbeddings, ChatOllama
class Models:
    def __init__(self):
        # ollama pull bge-m3 (multilingual embedding)
        self.embeddings_bge = OllamaEmbeddings(
            model="bge-m3"
        )

        # ollama pull neural-chat
        self.model_ollama = ChatOllama(
            model="mistral",
            temperature=0.1,
            
        )
        # ollama pull mxbai-embed-large
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # ollama pull llama3
        self.model_llama3 = ChatOllama(
            model="llama3",
            temperature=0.1,
            
        )