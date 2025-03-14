"""
Main module for the RAG application
"""
import logging
import nest_asyncio
from config import Config
from rag import create_knowledge_base, create_query_engine

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def query(query_engine, query_str)-> str:
    """
    Execute a query using the configured query engine.
    Args:
        query_engine: The query engine to use for executing the query.
        query_str: The query string to execute.
    Returns:
        str: The response from the query engine.
    """
    # Execute a query using the configured query engine.
    response = query_engine.query(query_str)
    logger.info(f"Response: {response}")
    return response


def main():
    """
    Main entry point for the application
    """
    logger.info("Starting RAG application")
    query_engine = setup_app()
    
    while True:
        query_str = input("Enter your query: ")
        response = query(query_engine, query_str)
        logger.info(f"Response: {response}")

def setup_app():
    """
    Setup the application by creating the knowledge base and query engine.
    """
    logger.info("Setting up application...")
    # Example of accessing environment variables
    if Config.DEBUG:
        logger.info("Debug mode is enabled")
    
    # Setup async
    # (LlamaIndex and other libraries that work with LLMs often use asynchronous operations internally, 
    # which might conflict if multiple async operations need to run simultaneously)
    nest_asyncio.apply()

    # Create the knowledge base from the input directory embedding in Qdrant the documents (of provided type) included in the directory
    vector_store_index = create_knowledge_base(Config.INPUT_DOC_FOLDER, 
                                               Config.DOC_FILE_TYPES, 
                                               Config.VECTOR_STORE_DOC_COLLECTION_NAME, 
                                               Config.QDRANT_HOST, 
                                               Config.QDRANT_PORT)

    # Create a query engine from the vector store index defined as knowledge base
    query_engine = create_query_engine(vector_store_index)
    logger.info("Application setup completed!")
    return query_engine
    
if __name__ == "__main__":
    main()
