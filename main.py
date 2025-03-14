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

def scan_docs(input_dir_path) -> List[Document]:
    """
    Scan the input directory for documents of the specified type.
    Args:
        input_dir_path: Path to the directory containing the documents.
    Returns:
        List[Document]: List of documents loaded from the input directory.
    """
    # llamaindex utility method that scans a directory, filters for specific file types, and loads document content into a format we can work with.
    directory_reader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            # Load specific types of files from the given directory...
            required_exts=[Config.DOC_FILE_TYPE],
            # ...recursively.
            recursive=True
        )
    # So far, the loader object has only been instantiated. We haven't read anything yet. 
    # Thus, the load_data() method is used to read the PDF file's content 
    # and return it in a structured format, storing it in docs list.
    docs = directory_reader.load_data()
    logger.info(f"Loaded {len(docs)} documents from:{input_dir_path}")
    return docs

def create_reranker() -> SentenceTransformerRerank:
    """
    Create a reranker for the given context and query.
    
    Returns:
        Reranker: The created reranker.
    """
    rerank = SentenceTransformerRerank(model=Config.RERANKER_MODEL, top_n=3)
    return rerank

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

    # Load the embedding model from the configuration and set it as the default embedding model in settings.
    load_embedding_model()
    # Load the LLM from the configuration and set it as the default LLM in settings.
    load_llm()

    # Initialize a QdrantClient instance, connecting it to the Qdrant server running locally.
    logger.info("Initializing QDRANT client")
    collection_name=Config.VECTOR_STORE_DOC_COLLECTION_NAME
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    logger.info("Qdrant client initialized successfully")

    input_dir_path = Config.INPUT_DOC_FOLDER
    docs = scan_docs(input_dir_path)
    
    # Create a vector store index for the given documents, adding to the given collection.
    # This function converts each document into an embedding using loaded embed_model
    # and stores the embeddings in Qdrant.
    vector_store_index = create_index(docs, client, collection_name)
    logger.info("Index created successfully")

    # Create a reranker for the given context and query.
    reranker = create_reranker()
    # Convert the previously created index into a query engine
    # - We specify that the engine should retrieve the top 10 most similar document chunks based on vector similarity to the query. 
    #   This forms the initial set of chunks for answering the query.
    # - The re-ranking step is further added to this to refine the retrieved chunks. 
    #   The rerank model will evaluate these chunks to select the most relevant ones for generating a response.
    query_engine = vector_store_index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])

    # Create a prompt template for the given context and query.
    qa_prompt_tmpl = create_prompt_template()
    # Add the prompt template to the query engine.
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    # Execute a query using the configured query engine.
    response = query_engine.query("Explain what is the design pattern fire and forget and how it works")
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()
