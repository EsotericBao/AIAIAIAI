import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from multiprocessing import Pool
import numpy as np
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import hashlib
from config import PDF_FOLDER, TEXTS_FOLDER, CHROMADB, COLLECTION, Models

    

def generate_hash_from_file(file_path):
    """
    Generate a deterministic hash based on the file's content.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):  # Read in chunks to handle large files
            hasher.update(chunk)
    return hasher.hexdigest()

def is_pdf_searchable(pdf_path):
    """
    Check if a PDF contains searchable text.
    Returns True if text is found, False otherwise.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            if doc.page_content.strip():  # If any meaningful text is found
                return True
    except Exception as e:
        print(f"Error checking if PDF is searchable: {e}")
    return False

def output_text(text, base_name, pdf_path):
    # Save extracted text to output file
    with open(os.path.join(TEXTS_FOLDER, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved for {pdf_path}")

# Chunk text using LangChain's RecursiveCharacterTextSplitter
def chunk_text(text, chunk_size=1000, chunk_overlap=50):
    """
    Split the text into smaller chunks for better embedding and retrieval performance.
    
    Args:
        text (str): The text to be chunked.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks to retain context.
    
    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# Process searchable pPDF
def process_searchable_pdf(pdf_path, output_dir,existing_hashes, chunk_size=1000, chunk_overlap=50):
    """
    Process searchable PDFs by extracting text directly and storing embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_hash = generate_hash_from_file(pdf_path)

    if file_hash in existing_hashes:
        print(f"Skipping searchable PDF: {pdf_path} (Hash already exists)")
        return

    print(f"Processing searchable PDF: {pdf_path} with hash: {file_hash}")
    
    loader = PyPDFLoader(pdf_path)
    loaded_documents = loader.load()
    # Combine all page contents into a single string
    full_text = "\n".join([doc.page_content for doc in loaded_documents if doc.page_content.strip()])
    # Save the extracted text to the output file
    output_text(full_text, base_name, pdf_path)


    # Chunk text and store in ChromaDB
    text_chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    filtered_chunks = [chunk for chunk in text_chunks if chunk.strip()]
    models = Models()
    embeddings = models.embeddings_ollama
    collection = Chroma(collection_name=COLLECTION,
                         embedding_function=embeddings, 
                         persist_directory=CHROMADB)
    
    collection.add_texts(
        texts=filtered_chunks,
        metadatas=[{"source": base_name, "hash": file_hash}] * len(text_chunks)
    )
    print(f"Finished processing searchable PDF: {pdf_path}")

def process_page(image):
    """
    Perform OCR on a single image (page).
    """
    # Initialize PaddleOCR (reinitialize for each process in multiprocessing)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Perform OCR
    results = ocr.ocr(image_np)
    page_text = " ".join([line[1][0] for line in results[0]])
    return page_text

# Extract text from a PDF
def process_scanned_pdf(pdf_path, output_dir,existing_hashes, dpi=300, use_multiprocessing=True, chunk_size=1000, chunk_overlap=50):
    """
    Process non-searchable (scanned) PDFs using OCR and store embeddings.
    """
    # Get the base name of the PDF file for naming the output text file
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_hash = generate_hash_from_file(pdf_path)

    print(f"Processing searchable PDF: {pdf_path} with Hash: {file_hash}")
    if file_hash in existing_hashes:
        print(f"Skipping scanned PDF: {pdf_path} (Hash already exists)")
        return
    
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)

    print(f"Extracting text from images in {pdf_path}...")
    if use_multiprocessing:
        # Multiprocessing: Each worker initializes PaddleOCR
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(process_page, images)
        extracted_text = "\n".join(results)
    else:
        # Sequential processing
        extracted_text = "\n".join([process_page(image) for image in images])

    # Combine the text from all pages
    full_text = "\n".join([f"Page {i+1}:\n{page_text}" for i, page_text in enumerate(extracted_text)])

    # Save the extracted text to the output file
    output_text(full_text, base_name, pdf_path)

    # Chunk text and store in ChromaDB
    text_chunks = chunk_text(extracted_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    filtered_chunks = [chunk for chunk in text_chunks if chunk.strip()]
    models = Models()
    embeddings = models.embeddings_ollama
    collection = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMADB,
    )

    collection.add_texts(
        texts=text_chunks,
        metadatas=[{"source": base_name, "hash": file_hash}] * len(text_chunks)
    )
    print(f"Finished processing scanned PDF: {pdf_path}")






# Process a folder of PDFs and store embeddings in ChromaDB
def process_folder(folder_path, output_dir, dpi=300, use_multiprocessing=True, chunk_size=1000, chunk_overlap=50):
    """
    Process all PDFs in a folder and store text embeddings in ChromaDB, dynamically determining if they are searchable or scanned, skipping duplicates.
    """

    models = Models()
    embeddings = models.embeddings_ollama
    collection = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMADB,
    )

    existing_hashes = set(metadata.get("hash") for metadata in collection.get()["metadatas"] if "hash" in metadata)
    print(f"Existing hashes in ChromaDB: {existing_hashes}")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            if is_pdf_searchable(pdf_path):
                process_searchable_pdf(pdf_path, output_dir, existing_hashes, chunk_size, chunk_overlap)
            else:
                process_scanned_pdf(pdf_path, output_dir, existing_hashes, dpi, use_multiprocessing, chunk_size, chunk_overlap)


if __name__ == "__main__":
    # Folder containing PDF files
    folder_path_scanned = os.path.join(PDF_FOLDER, "scanned")
    folder_path_searchable = os.path.join(PDF_FOLDER, "searchable")
    output_dir = TEXTS_FOLDER
    # ChromaDB persistence directory
    persist_directory = CHROMADB

    # Process folder and store embeddings
    process_folder(
        PDF_FOLDER,
        output_dir,
        chunk_size=1000,
        chunk_overlap=50,
    )