"""
Configuration utility module that loads environment variables
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file
env_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration class to access environment variables
class Config:
    """Configuration class that provides access to environment variables"""
    
    # PDF document folder
    INPUT_DOC_FOLDER = os.getenv('INPUT_DOC_FOLDER', './data') 

    # Document file extension to process
    DOC_FILE_TYPES = os.getenv('DOC_FILE_TYPES', '.pdf').split(',')

    # Vector store collection name
    VECTOR_STORE_DOC_COLLECTION_NAME = os.getenv('VECTOR_STORE_DOC_COLLECTION_NAME')

    # The model to use for embedding
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    
    # The LLM model to use for generation
    LLM = os.getenv('LLM')
    
    # The model to use for reranking
    RERANKER_MODEL = os.getenv('RERANKER_MODEL')
    
    # Qdrant settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '6333'))
    
    # LLM settings
    LLM_REQ_TIMEOUT_SECONDS = float(os.getenv('LLM_REQ_TIMEOUT_SECONDS', '120.0'))
    
    # Application settings
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
