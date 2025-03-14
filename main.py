"""
Main module for the RAG application
"""
import logging

from llama_index.core.instrumentation.span_handlers import null
from config import Config
import nest_asyncio
import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import Document
from llama_index.core.query_engine import BaseQueryEngine
from typing import List

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_embedding_model() -> HuggingFaceEmbedding:
    """
    Load the embedding model from the configuration 
    and set it as the default embedding model in settings.
    """
    embed_model = HuggingFaceEmbedding(model_name=Config.EMBEDDING_MODEL, trust_remote_code=True)

    # Configure embed_model as the default embedding model in Settings. 
    # (This setting ensures that the same model is used throughout our RAG pipeline 
    # to maintain consistency in embedding generation.)
    Settings.embed_model = embed_model
    logger.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")
    return embed_model

def load_llm():
    """
    Load the LLM from the configuration and set it as the default LLM in settings.
    Args:
        None
    Returns:
        Ollama: The loaded LLM instance.
    """
    # Specifying a request_timeout of 120 seconds 
    # for requests to the LLM to ensure that the system doesn't get stuck 
    # if the model takes too long to respond.
    llm = Ollama(model=Config.LLM, request_timeout=120.0)
    # Set the above LLM instance as the default language model in Settings, 
    # making it the *primary model* used in our RAG pipeline.
    Settings.llm = llm
    logger.info(f"Loaded LLM: {Config.LLM}")
    return llm

def create_index(documents, client, collection_name):
    """
    Create a vector store index for the given documents, adding to the given collection.
    
    Args:
        documents: List of documents to add to the index.
        client: Qdrant client instance.
        collection_name: Name of the collection to add the documents to.
    """
    # Initialize a QdrantVectorStore object by passing the previously created Qdrant client and a name for the collection.
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    # Configure storage settings by specifying the above vector_store as the storage backend.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Create an index by embedding each document in documents and storing it in the Qdrant vector store
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def create_prompt_template():
    """
    Create a prompt template for the given context and query.
    Args:
        context_str: The context information to be used in the prompt.
        query_str: The query to be used in the prompt.
    Returns:
        PromptTemplate: The created prompt template.
    """
    template = """Context information is below:
              ---------------------
              {context_str}
              ---------------------
              Given the context information above I want you to think
              step by step to answer the query in a crisp manner.
              Cite always the source of the information you are using.
              In case you don't know the answer say 'I don't know!'
            
              Query: {query_str}
        
              Answer:"""

    qa_prompt_tmpl = PromptTemplate(template)
    return qa_prompt_tmpl

def scan_docs(input_dir_path: str, file_type_list: list[str]) -> List[Document]:
    """
    Scan the input directory for documents of the specified type.
    Args:
        input_dir_path: Path to the directory containing the documents.
        file_type_list: List of file types to load from the directory.
    Returns:
        List[Document]: List of documents loaded from the input directory.
    """
    logger.info(f"Scanning directory:{input_dir_path} for files of type:{file_type_list}")
    # llamaindex utility method that scans a directory, filters for specific file types, and loads document content into a format we can work with.
    directory_reader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            # Load specific types of files from the given directory...
            required_exts=file_type_list,
            # ...recursively.
            recursive=True
        )
    # So far, the loader object has only been instantiated. We haven't read anything yet. 
    # Thus, the load_data() method is used to read the PDF file's content 
    # and return it in a structured format, storing it in docs list.
    docs = directory_reader.load_data()
    logger.info(f"Loaded {len(docs)} documents from:{input_dir_path} of type:{file_type_list}")
    return docs

def create_reranker() -> SentenceTransformerRerank:
    """
    Create a reranker for the given context and query.
    
    Returns:
        Reranker: The created reranker.
    """
    # top_n: Number of top results to return after reranking; limits the final results to only the 3 most relevant chunks
    rerank = SentenceTransformerRerank(model=Config.RERANKER_MODEL, top_n=3)  
    return rerank

def create_knowledge_base(input_dir_path: str, file_type_list: list[str], collection_name: str, qdrant_host: str, qdrant_port: int) -> VectorStoreIndex:
    """
    Create a knowledge base from the input directory embedding in Qdrant the documents (of provided type) included in the directory.
    Args:
        input_dir_path: Path to the directory containing the documents.
        file_type_list: List of file types to load from the directory.
        collection_name: Name of the collection to add the documents to.
        qdrant_host: Hostname of the Qdrant server.
        qdrant_port: Port number of the Qdrant server.
    """
    # Load the embedding model from the configuration and set it as the default embedding model in settings.
    load_embedding_model()

    # Initialize a QdrantClient instance, connecting it to the Qdrant server running locally.
    logger.info("Initializing QDRANT client")
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Qdrant client initialized successfully at {qdrant_host}:{qdrant_port}")
    # Scan the provided input directory for documents of the specified type.
    docs = scan_docs(input_dir_path, file_type_list)
    
    # Create a vector store index for the given documents, adding to the given collection.
    # This function converts each document into an embedding using loaded embed_model
    # and stores the embeddings in Qdrant.
    vector_store_index = create_index(docs, client, collection_name)
    logger.info("Index created successfully")
    return vector_store_index

def create_query_engine(vector_store_index: VectorStoreIndex) -> BaseQueryEngine:
    # Create a reranker for the given context and query.
    reranker = create_reranker()
    # Create a prompt template for the given context and query.
    qa_prompt_tmpl = create_prompt_template()
    # Load the LLM from the configuration and set it as the default LLM in settings.
    load_llm()

    # Convert the previously created index into a query engine
    # - We specify that the engine should retrieve the top 10 most similar document chunks based on vector similarity to the query. 
    #   This forms the initial set of chunks for answering the query.
    # - The re-ranking step is further added to this to refine the retrieved chunks. 
    #   The rerank model will evaluate these chunks to select the most relevant ones for generating a response.
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])

    # Add the prompt template to the query engine.
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
    return query_engine

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

def setup_app()-> BaseQueryEngine:
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
