# RAG System Implementation

A practical implementation of a Retrieval-Augmented Generation (RAG) system, inspired by the article [A Crash Course on Building RAG Systems (Part 1)](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/).

## Overview

This project implements a RAG system that enhances Large Language Model (LLM) responses by retrieving relevant information from a document corpus. The system processes PDF documents, embeds them into vector representations, stores them in a vector database (Qdrant), and uses them to provide context-aware responses to user queries.

## Architecture

The system consists of the following components:

1. **Document Processing**: Uses LlamaIndex to load and process PDF documents
2. **Vector Database**: Qdrant for efficient storage and retrieval of document embeddings
3. **Embedding Model**: Converts documents and queries into vector embeddings
4. **Language Model**: Generates responses based on retrieved context and user queries
5. **Configuration Management**: Handles environment variables and project settings

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for running Qdrant)
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
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `INPUT_DOC_FOLDER`: Directory containing your PDF documents (default: `./data`)
     - Other database and application settings

4. **Start the Qdrant vector database**:
   ```bash
   docker-compose up -d
   ```
   The Qdrant dashboard will be available at http://localhost:6333/dashboard

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
├── main.py           # Main application entry point
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
   - Document chunks are embedded into vector representations

3. **Storage**:
   - Embeddings are stored in the Qdrant vector database for efficient retrieval

4. **Query Processing**:
   - User queries are embedded using the same embedding model
   - Similar document chunks are retrieved from Qdrant based on vector similarity
   - Retrieved information is used to provide context to the LLM

5. **Response Generation**:
   - The LLM generates a response that incorporates both the user query and the retrieved context

## Further Development

- Implement a web interface for interacting with the system
- Add support for more document types (beyond PDFs)
- Implement query-specific optimizations
- Add evaluation metrics for measuring system performance

## References

- [A Crash Course on Building RAG Systems (Part 1)](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)