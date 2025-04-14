import os
import hashlib
import logging
import numpy as np
import re
import unicodedata
from pathlib import Path
from typing import List, Set

from functools import lru_cache

from dataclasses import dataclass
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import TEXTS_FOLDER, MAIN_FOLDER, MAINDB, COMMON_PDF, CHROMADB, FALLBACK_PDF, FALLBACKDB, COLLECTION, Models, MAIN_COLLECTION, FALLBACK_COLLECTION

# Configure Logging
# logger = logging.getLogger("__name__")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

@dataclass
class ChromaConfig:
    """Configuration for ChromaDB ingestion."""
    collection_name: str
    persist_directory: str
    input_folder: str

class DocumentProcessor:
    """Handles document ingestion, OCR, and embedding storage."""
    def __init__(self, config: ChromaConfig, models):

        self.config = config
        self.models = models
        self.embeddings = models.embeddings_main
        self.collection = self._initialize_collection()

        # Ensure input folder exists
        self.ensure_folder_exists(self.config.input_folder)
        #self.ensure_folder_exists(TEXTS_FOLDER)

    def _initialize_collection(self) -> Chroma:
        """Initialize ChromaDB collection."""
        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory
        )
    
    @staticmethod
    def ensure_folder_exists(folder_path: str):
        """Ensure folder exists, create if missing."""
        path = Path(folder_path)
        if not path.exists():
            # logger.warning
            print(f"⚠️ Folder {folder_path} does not exist. Creating it...")
            path.mkdir(parents=True, exist_ok=True)
        else:
            # logger.info
            print(f"✅ Folder exists: {folder_path}")

    @staticmethod
    @lru_cache(maxsize=1000)
    def generate_hash_from_file(file_path: str) -> str:
        """Generate a unique hash from file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_existing_hashes(self) -> Set[str]:
        """Retrieve existing document hashes from ChromaDB."""
        return {
            metadata.get("hash")
            for metadata in self.collection.get()["metadatas"]
            if metadata and "hash" in metadata
        }

    @staticmethod
    def determine_chunk_size(text: str) -> tuple:
        """Dynamically determine chunk size based on text length."""
        word_count = len(text.split())
        if word_count < 500:
            return 300, 50
        elif word_count < 2000:
            return 800, 100
        return 1500, 200
    
    def clean_text(self, text: str) -> str:
        """Cleans and normalizes text before chunking."""
        # Normalize Unicode characters (e.g., replace accented characters)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r"\s+", " ", text).strip()

        # Remove page numbers (if detected as standalone numbers)
        text = re.sub(r"\n\d+\n", "\n", text)
        text = re.sub(r"\bPage\s*\d+:\s*", "", text)    # Removes "Page X"

        # Remove common boilerplate text patterns (optional)
        boilerplate_patterns = [
            r"Copyright ©.*?\d{4}",  # Remove copyright lines
            r"Page \d+ of \d+",  # Remove "Page X of Y" patterns
            r"Page \d+ :",  #Remove "Page X"
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into dynamically-sized chunks."""
        # Clean the text before chunking
        cleaned_text = self.clean_text(text)

        chunk_size, chunk_overlap = self.determine_chunk_size(cleaned_text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return [chunk for chunk in splitter.split_text(cleaned_text) if chunk.strip()]

    def save_text(self, text, base_name: str) -> None:
        """Save extracted text to a file."""
        text_path = Path(TEXTS_FOLDER) / f"{base_name}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("<🪓>".join(text), encoding="utf-8")
        # logger.info
        print(f"Text saved to {text_path}")

    def process_text_file(self, file_path: str, existing_hashes: Set[str]) -> None:
        """Process and ingest plain text files."""
        base_name = Path(file_path).stem
        file_hash = self.generate_hash_from_file(file_path)

        if file_hash in existing_hashes:
            # logger.info
            print(f"🟡Skipping text file {file_path} (Already processed)")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read().strip()

            if not full_text:
                # logger.warning
                print(f"Empty text file: {file_path}")
                return

            
            text_chunks = self.chunk_text(full_text)
            self.save_text(text_chunks, base_name)

            self.collection.add_texts(
                texts=text_chunks,
                metadatas=[{"source": base_name, "hash": file_hash}] * len(text_chunks)
            )
            # logger.info
            print(f"Processed text file: {file_path}")

        except Exception as e:
            # logger.error
            print(f"🔴Error processing text file {file_path}: {str(e)}")

    def process_searchable_pdf(self, pdf_path: str, existing_hashes: Set[str]) -> None:
        """Process PDFs with embedded text."""
        base_name = Path(pdf_path).stem
        file_hash = self.generate_hash_from_file(pdf_path)

        if file_hash in existing_hashes:
            # logger.info
            print(f"🟡Skipping {pdf_path} (Already processed)")
            return

        try:
            print(f"🟢Processing searchable PDF: {pdf_path} with hash: {file_hash}")
            loader = PyPDFLoader(pdf_path)
            full_text = "\n".join(
                doc.page_content
                for doc in loader.load()
                if doc.page_content.strip()
            )

            if not full_text.strip():
                # logger.warning
                print(f"No text extracted from {pdf_path}")
                return

            
            text_chunks = self.chunk_text(full_text)
            self.save_text(text_chunks, base_name)
            self.collection.add_texts(
                texts=text_chunks,
                metadatas=[{"source": base_name, "hash": file_hash}] * len(text_chunks)
            )
            # logger.info
            print(f"Processed: {pdf_path}")

        except Exception as e:
            # logger.error
            print(f"🔴Error processing {pdf_path}: {str(e)}")

    def process_scanned_pdf(self, pdf_path: str, existing_hashes: Set[str]) -> None:
        """Process scanned PDFs using OCR."""
        base_name = Path(pdf_path).stem
        file_hash = self.generate_hash_from_file(pdf_path)

        if file_hash in existing_hashes:
            # logger.info
            print(f"Skipping {pdf_path} (Already processed)")
            return

        try:
            extracted_text = self.process_page(pdf_path)
            
            text_chunks = self.chunk_text(extracted_text)
            self.save_text(text_chunks, base_name)
            self.collection.add_texts(
                texts=text_chunks,
                metadatas=[{"source": base_name, "hash": file_hash}] * len(text_chunks)
            )
            # logger.info
            print(f"🟢Processing PDF: {pdf_path} with hash: {file_hash}")

        except Exception as e:
            # logger.error
            print(f"🔴Error processing {pdf_path}: {str(e)}")

    @staticmethod
    def process_page(pdf) -> str:
        """Extract text from an image using OCR."""
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
        results = ocr.ocr(pdf)
        return " ".join(line[1][0] for line in results[0]) if results[0] else ""

    def process_folder(self) -> None:
        """Process all PDFs in the configured folder."""
        existing_hashes = self.get_existing_hashes()
        input_folder = Path(self.config.input_folder)

        if not input_folder.exists():
            # logger.error
            print(f"🔴Folder not found: {input_folder}")
            return

        for file_path in input_folder.glob("**/*"):
            try:
                if file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    is_searchable = bool(loader.load()[0].page_content.strip())

                    if is_searchable:
                        self.process_searchable_pdf(str(file_path), existing_hashes)
                    else:
                        self.process_scanned_pdf(str(file_path), existing_hashes)
                elif file_path.suffix.lower() == ".txt":
                    self.process_text_file(str(file_path), existing_hashes)

            except Exception as e:
                # logger.error
                print(f"🔴Error processing {file_path}: {str(e)}")

def ingest_documents(config: ChromaConfig, models) -> None:
    """Main ingestion function."""
    try:
        # logger.info
        print(f"Starting ingestion for {config.collection_name}")
        processor = DocumentProcessor(config, models)
        processor.process_folder()
        # logger.info
        print(f"Completed ingestion for {config.collection_name}")
    except Exception as e:
        # logger.error
        print(f"🔴Ingestion failed for {config.collection_name}: {str(e)}")

if __name__ == "__main__":
    configs = {
        "main": ChromaConfig(MAIN_COLLECTION, MAINDB, MAIN_FOLDER),
        #"common": ChromaConfig(COLLECTION, CHROMADB, COMMON_PDF),
        #"fallback": ChromaConfig(FALLBACK_COLLECTION, FALLBACKDB, FALLBACK_PDF)
    }

    models = Models()
    print("start")
    for name, config in configs.items():
        # logger.info
        print(f"Starting {name} ingestion...")
        ingest_documents(config, models)
        # logger.info
        print(f"Completed {name} ingestion")
