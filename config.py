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
    DOC_FILE_TYPE = os.getenv('DOC_FILE_TYPE', '.pdf')

    # Vector store collection name
    VECTOR_STORE_DOC_COLLECTION_NAME = os.getenv('VECTOR_STORE_DOC_COLLECTION_NAME', 'chat_with_docs')

    # The model to use for embedding
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    
    # The LLM model to use for generation
    LLM = os.getenv('LLM', 'llama3.2:1b')
    
    # The model to use for reranking
    RERANKER_MODEL = os.getenv('RERANKER_MODEL')
    # Database configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'rag_database')
    DB_USER = os.getenv('DB_USER', 'username')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    # Application settings
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_database_url(cls):
        """Returns the database URL constructed from individual settings"""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
