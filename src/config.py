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

    # API keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
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
