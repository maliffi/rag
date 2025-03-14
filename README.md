# RAG System Implementation

A practical implementation of a Retrieval-Augmented Generation (RAG) system, inspired by the article [A Crash Course on Building RAG Systems (Part 1)](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/).

## Overview

This project implements a RAG system that enhances Large Language Model (LLM) responses by retrieving relevant information from a document corpus. The system processes PDF documents, embeds them into vector representations, stores them in a vector database (Qdrant), and uses them to provide context-aware responses to user queries.

## Architecture

The system consists of the following components:

1. **Document Processing**: Uses LlamaIndex's SimpleDirectoryReader to load and process PDF documents
2. **Vector Database**: Qdrant for efficient storage and retrieval of document embeddings
3. **Embedding Model**: HuggingFace embedding model (default: 'BAAI/bge-large-en-v1.5') that converts documents and queries into vector embeddings
4. **Reranker**: SentenceTransformerRerank to improve retrieval quality by re-ranking the initial vector similarity results
5. **Language Model**: Ollama for generating responses based on retrieved context and user queries
6. **Configuration Management**: Handles environment variables and project settings through a Config class

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for running Qdrant)
- Ollama installed locally (for LLM inference)
- PDF documents to process

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your specific configuration:
     - `LOG_LEVEL`: Logging level (e.g., INFO, DEBUG)
     - `DEBUG`: Enable/disable debug mode
     - `LLM`: Ollama model to use (default: 'llama3.2:1b')
     - `EMBEDDING_MODEL`: HuggingFace embedding model (default: 'BAAI/bge-large-en-v1.5')
     - `RERANKER_MODEL`: SentenceTransformer reranker model name
     - `INPUT_DOC_FOLDER`: Directory containing your PDF documents (default: `./data`)
     - `DOC_FILE_TYPES`: Type of documents to process (default: pdf)
     - `VECTOR_STORE_DOC_COLLECTION_NAME`: Name for the Qdrant collection

4. **Start the Qdrant vector database**:
   ```bash
   docker-compose up -d
   ```
   The Qdrant dashboard will be available at http://localhost:6333/dashboard

5. **Install and run Ollama**:
   Follow the installation instructions at [Ollama's website](https://ollama.ai/) to set up Ollama on your machine.

## Usage

1. **Prepare your documents**:
   - Place your PDF documents in the folder specified by `INPUT_DOC_FOLDER`

2. **Run the application**:
   ```bash
   python main.py
   ```

## Project Structure

```
rag/
├── data/                 # Directory for PDF documents
├── main.py               # Main application entry point
├── config.py             # Configuration management
├── qdrant_storage/       # Storage for Qdrant (created by Docker)
├── docker-compose.yml    # Docker configuration for Qdrant
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## How it Works

1. **Document Ingestion**:
   - The system reads PDF documents using LlamaIndex's SimpleDirectoryReader
   - Documents are divided into chunks and processed

2. **Vector Embedding**:
   - Document chunks are embedded into vector representations using HuggingFace embedding models

3. **Storage**:
   - Embeddings are stored in the Qdrant vector database for efficient retrieval

4. **Query Processing**:
   - User queries are embedded using the same embedding model
   - Similar document chunks are retrieved from Qdrant based on vector similarity
   - A reranker (SentenceTransformerRerank) is applied to improve the relevance of retrieved chunks

5. **Response Generation**:
   - The Ollama LLM generates a response that incorporates both the user query and the retrieved context
   - A custom prompt template guides the LLM to provide accurate, well-cited responses

## Further Development

- Implement a web interface for interacting with the system
- Add support for more document types (beyond PDFs)
- Implement query-specific optimizations
- Add evaluation metrics for measuring system performance
- Explore different embedding models and reranker configurations

## References

- [A Crash Course on Building RAG Systems (Part 1)](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [HuggingFace Documentation](https://huggingface.co/docs)