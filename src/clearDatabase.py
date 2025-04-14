import os
import shutil
from config import COLLECTION, SPFDB, TEXTS_FOLDER, Models
from langchain_chroma import Chroma


def clear_directory(directory_path):
    """
    Clear all files and subdirectories in the specified directory.
    """
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"Cleared contents of {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")

chroma_config = {
    "collection_name" : COLLECTION,
    "persist_directory" : SPFDB,
    #"persist_directory" : CHROMADB,
}

# Clear the ChromaDB directory
shutil.rmtree(chroma_config["persist_directory"], ignore_errors=True)
# Clear the output text folder
#clear_directory(TEXTS_FOLDER)
shutil.rmtree(TEXTS_FOLDER, ignore_errors=True)


# Optional: Reinitialize the Chroma collection
models = Models()
embeddings = models.embeddings_spf
collection = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=chroma_config["persist_directory"],
)


print("Cleared!")
