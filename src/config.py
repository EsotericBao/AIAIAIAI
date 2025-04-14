import os

# Define base directories
SRC_DIR = os.path.abspath(os.path.dirname(__file__))  # Directory of the config file
BASE_DIR = os.path.dirname(SRC_DIR)
#PDF folders
PDF_FOLDER = os.path.join(SRC_DIR, "PDFs")
MAIN_FOLDER = os.path.join(PDF_FOLDER, "main_pdfs")
MULTI_PDF = os.path.join(PDF_FOLDER, "multilingual_pdfs")
COMMON_PDF = os.path.join(PDF_FOLDER, "general_pdfs")
FALLBACK_PDF = os.path.join(PDF_FOLDER, "fallback_pdfs")
#Text folders
TEXTS_FOLDER = os.path.join(SRC_DIR, "output_texts")

#Vector Database folders
VECTORDB = os.path.join(SRC_DIR, "vector_database")
CHROMADB = os.path.join(VECTORDB, "chromadb")
MAINDB = os.path.join(VECTORDB, "maindb")
FALLBACKDB = os.path.join(VECTORDB, "fallbackdb")
TESTDB = os.path.join(SRC_DIR, "testdb")
MAIN_COLLECTION ="main_embeddings"
COLLECTION ="general_embeddings"
FALLBACK_COLLECTION ="fallback_embeddings"


#Recordings folder
RECORDINGS_FOLDER = os.path.join(SRC_DIR, "recordings")

#Keys
GOOGLE_CLOUD_STT_KEY = os.path.join(BASE_DIR, "keys/decoded-academy-448509-g0-3a0140a8afb5.json")

#
#
#
#Set up llm models and embedding models
from langchain_ollama import OllamaEmbeddings, ChatOllama
class Models:
    def __init__(self):
        self.embeddings_main = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        self.model_main = ChatOllama(
            #model="mistral",
             #model="deepseek-r1:14b",
            #model="deepseek-r1:8b",
            #model="wizardlm2:7b",
            model="gemma3:4b",
            streaming=True,
            temperature=0,
            
        )
        # ollama pull mxbai-embed-large
        self.embeddings_large = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # ollama pull llama3
        self.model_fallback = ChatOllama(
            #model="mistral",
            model="wizardlm2:7b",
            temperature=0.1,
            
        )