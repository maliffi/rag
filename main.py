"""
Main module for the RAG application
"""
import logging
import nest_asyncio
from config import Config
from rag import create_knowledge_base, create_query_engine
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

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
    return response


def main():
    """
    Main entry point for the application
    """
    logger.info("Starting RAG application")
    query_engine = setup_app()

    # Load test data and convert to a pandas dataframe
    csv_file = 'testset/test_data_paul_graham.csv'
    test_df = pd.read_csv(csv_file).dropna()

    # Extract questions  
    test_questions = test_df["question"].values

    # Generate a response for each question
    responses = [generate_response(query_engine, q) for q in tqdm(test_questions)]

    # Build a dataset dict putting together questions, answers, contexts and ground truth
    dataset_dict = {
        "question": test_questions,
        "answer": [response["answer"] for response in responses],
        "contexts": [response["contexts"] for response in responses],
        "ground_truth": test_df["ground_truth"].values.tolist(),
    }

    # Build a dataset from the dict
    ragas_eval_dataset = Dataset.from_dict(dataset_dict)

    # Recall that we need a Critic LLM along with an embedding model to compute similarities when needed.
    critic_llm = OllamaLLM(model="llama3.2:1b")
    ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

    metrics = [faithfulness, answer_correctness, context_recall, context_precision]
    # Increase timeout and add retry count to handle parsing errors
    run_config = RunConfig(timeout=6000, max_retries=10)  
    # Now, we can again use Ragas to compute the metrics
    evaluation_result = evaluate(
        llm=critic_llm,
        embeddings=ollama_emb,
        dataset=ragas_eval_dataset,
        metrics=metrics,
        run_config=run_config
    )
    # The evaluation_results object can be viewed as a Pandas DataFrame
    eval_scores_df = pd.DataFrame(evaluation_result.scores)
    logger.info("Evaluation scores:\n", eval_scores_df)

def generate_response(query_engine, question) -> dict:
    """
    Generate a response to a question using the configured query engine.
    
    Args:
        query_engine: The query engine to use for generating the response.
        question: The question to generate a response for.
    Returns:
        dict: A dictionary containing the generated response and the source nodes.
    """
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }

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
                                               Config.DB_HOST, 
                                               Config.DB_PORT)

    # Create a query engine from the vector store index defined as knowledge base
    query_engine = create_query_engine(vector_store_index)
    logger.info("Application setup completed!")
    return query_engine
    
if __name__ == "__main__":
    main()
