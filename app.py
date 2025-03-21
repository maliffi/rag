"""
Streamlit web application for the RAG system
"""

import os
import tempfile
import streamlit as st
import logging
from pathlib import Path

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import Document

from config import Config
from rag import (
    load_embedding_model,
    load_llm,
    scan_docs,
    create_index,
    create_prompt_template,
    create_reranker
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BF5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5C5C5C;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    .response-container {
        padding: 20px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_components():
    """Initialize the RAG components and return them"""
    # Load models
    embed_model = load_embedding_model()
    llm = load_llm()
    
    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(host=Config.DB_HOST, port=Config.DB_PORT)
    
    # Create reranker
    reranker = create_reranker()
    
    # Create prompt template
    qa_prompt_tmpl = create_prompt_template()
    
    return {
        "embed_model": embed_model,
        "llm": llm,
        "client": client,
        "reranker": reranker,
        "qa_prompt_tmpl": qa_prompt_tmpl
    }

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return the path"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Get file extension
    file_ext = Path(uploaded_file.name).suffix
    
    # Save the file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_dir, file_path

def main():
    # Display header
    st.markdown("<h1 class='main-header'>📚 RAG Document Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload documents, ask questions, get answers</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = Config.VECTOR_STORE_DOC_COLLECTION_NAME
    
    # Initialize RAG components
    rag_components = initialize_rag_components()
    
    # Sidebar for configuration and document upload
    with st.sidebar:
        st.header("Configuration")
        
        # Display current configuration
        st.subheader("Current Settings")
        st.write(f"Embedding Model: `{Config.EMBEDDING_MODEL}`")
        st.write(f"LLM: `{Config.LLM}`")
        st.write(f"Reranker Model: `{Config.RERANKER_MODEL}`")
        st.write(f"Collection Name: `{st.session_state.collection_name}`")
        
        # Option to change collection name
        new_collection = st.text_input(
            "Change Collection Name (optional)",
            value="",
            placeholder="Enter a new collection name"
        )
        if new_collection:
            st.session_state.collection_name = new_collection
            st.success(f"Collection name changed to: {new_collection}")
        
        st.divider()
        
        # Document upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Index Documents"):
                with st.spinner("Indexing documents..."):
                    # Save uploaded files to a temporary directory
                    temp_dirs = []
                    documents = []
                    
                    for uploaded_file in uploaded_files:
                        temp_dir, file_path = save_uploaded_file(uploaded_file)
                        temp_dirs.append(temp_dir)
                        
                        # Extract file extension from file_path
                        file_ext = os.path.splitext(file_path)[1]  # Keep the dot in extension
                        logger.info(f"File extension: {file_ext}")
                        # Load the saved document with specific file extension
                        file_docs = scan_docs(temp_dir, [file_ext])
                        documents.extend(file_docs)
                    
                    if documents:
                        # Create index
                        vector_store_index = create_index(
                            documents,
                            rag_components["client"],
                            st.session_state.collection_name
                        )
                        
                        # Create query engine
                        query_engine = vector_store_index.as_query_engine(
                            similarity_top_k=10,
                            node_postprocessors=[rag_components["reranker"]]
                        )
                        
                        # Add prompt template
                        query_engine.update_prompts({
                            "response_synthesizer:text_qa_template": rag_components["qa_prompt_tmpl"]
                        })
                        
                        # Save query engine to session state
                        st.session_state.query_engine = query_engine
                        
                        st.success(f"Successfully indexed {len(documents)} documents!")
                    else:
                        st.error("No documents were successfully loaded.")
    
    # Main area for Q&A
    st.header("Ask Questions")
    
    # Check if documents have been indexed
    if st.session_state.query_engine is None:
        st.info("Please upload and index documents using the sidebar before asking questions.")
    else:
        # Display conversation history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**Assistant:** {response}")
            st.divider()
        
        # Question input
        query = st.text_input("Ask a question about your documents:", key="question_input")
        
        if query and "last_query" not in st.session_state:
            st.session_state.last_query = query
            with st.spinner("Generating answer..."):
                try:
                    # Execute query
                    response = st.session_state.query_engine.query(query)
                    
                    # Display response
                    st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                    st.markdown(f"**Answer:** {response}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((query, str(response)))
                    
                    # Clear the last_query to allow new questions
                    # This approach avoids the need for experimental_rerun()
                    if "last_query" in st.session_state:
                        del st.session_state.last_query
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if "last_query" in st.session_state:
                        del st.session_state.last_query
    
    # App footer
    st.divider()
    st.caption("RAG Document Assistant - Powered by LlamaIndex, Qdrant, and Ollama. Source code written by Matteo Aliffi.")

if __name__ == "__main__":
    main()
