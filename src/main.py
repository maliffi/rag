"""
Main module for the RAG application
"""
import logging
from config import Config
import nest_asyncio
import qdrant_client

from llama_index.core import SimpleDirectoryReader

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the application
    """
    logger.info("Starting RAG application")
    
    # Example of accessing environment variables
    if Config.DEBUG:
        logger.info("Debug mode is enabled")
    
    # Setup async
    nest_asyncio.apply()

    # Initialize a QdrantClient instance, connecting it to the Qdrant server running locally.
    collection_name="chat_with_docs"
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    logger.info("Qdrant client initialized successfully")

    input_dir_path = Config.INPUT_DOC_FOLDER
    # Scan a directory, filters for specific file types, and loads document content into a format we can work with.
    loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            # Load specific types of files (.pdf) from the given directory...
            required_exts=[".pdf"],
            # ...recursively.
            recursive=True
        )
    # So far, the loader object has only been instantiated. We haven't read anything yet. 
    # Thus, the load_data() method is used to read the PDF file’s content 
    # and return it in a structured format, storing it in docs list.
    docs = loader.load_data()
    logger.info(f"Loaded {len(docs)} documents from:{input_dir_path}")


if __name__ == "__main__":
    main()
