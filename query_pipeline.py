import os
from dotenv import load_dotenv
import chromadb
import openai
import logging
from typing import List, Dict, Any
from pathlib import Path
from prompts import SYSTEM_MESSAGE, USER_MESSAGE_TEMPLATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "agent-converto")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_PATH = Path("./chroma_db")
# You might want a different LLM model for generation, e.g., "gpt-4", "gpt-3.5-turbo"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
N_SEARCH_RESULTS = 5 # Number of relevant chunks to retrieve from the DB

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY", "NOTION_TOKEN", "DATABASE_ID"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")
    # Exit if essential variables are missing for querying
    exit(1)


def get_embedding(text: str, model_name: str = OPENAI_EMBEDDING_MODEL) -> List[float]:
    """
    Get embedding for a single text using OpenAI API.
    """
    try:
        logger.info(f"Requesting embedding for text snippet...")
        response = openai.embeddings.create(
            model=model_name,
            input=[text]
        )
        # Assuming response.data is a list with one item for a single input
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("OpenAI embeddings API returned empty data.")
    except Exception as e:
        logger.error(f"Error getting embedding for query: {str(e)}")
        raise

def search_database(query_embedding: List[float], n_results: int = N_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for relevant documents.
    """
    logger.info(f"Searching ChromaDB collection '{VECTOR_DB_COLLECTION}' for top {n_results} results...")
    try:
        # Initialize ChromaDB with persistence
        if not CHROMA_DB_PATH.exists():
             logger.error(f"ChromaDB path does not exist: {CHROMA_DB_PATH}. Cannot perform search.")
             return []

        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

        # Get the collection
        try:
            # Use get_collection if you expect it to exist, or get_or_create_collection if you're unsure
            # Since embed_pipeline creates it, get_collection is appropriate here
            collection = client.get_collection(VECTOR_DB_COLLECTION)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas'] # Include text content and metadata
            )
            
            # The results structure is nested, extract the relevant parts
            # Assuming a single query_embedding, results['ids'][0], results['documents'][0], etc.
            extracted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    extracted_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })

            return extracted_results

        except Exception as e:
            logger.error(f"Error accessing ChromaDB collection '{VECTOR_DB_COLLECTION}' during search: {e}")
            return []

    except Exception as e:
        logger.error(f"Error initializing ChromaDB client during search: {e}")
        return []

def format_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Formats the search results into a string to be used as context for the LLM.
    """
    if not search_results or (len(search_results) == 1 and "No relevant information" in search_results[0].get('document', '')):
        return "No relevant information found in the knowledge base."
        
    context_string = "Using the following document excerpts as context:\n---\n"
    for i, result in enumerate(search_results):
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'Unknown Title')
        course = metadata.get('course', 'Unknown Course')
        document = result.get('document', 'No content available')
        
        context_string += f"Source {i+1} (Course: {course}, Title: {title}):\n{document}\n---\n"
        
    return context_string

def generate_answer(query: str, context: str, model_name: str = LLM_MODEL) -> str:
    """
    Generates an answer to the query using the provided context and an LLM.
    """
    logger.info(f"Generating answer using LLM model '{model_name}'...")
    try:
        # Construct the prompt for the LLM using templates from prompts.py
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE_TEMPLATE.format(context=context, query=query)}
        ]

        response = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7, # Adjust temperature for creativity (lower for more factual)
            max_tokens=500 # Adjust max_tokens as needed
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "The LLM did not return a response."

    except Exception as e:
        logger.error(f"Error generating answer with LLM: {str(e)}")
        return f"An error occurred while generating the answer: {str(e)}"

if __name__ == "__main__":
    # Simple command-line input for demonstration
    user_query = input("Enter your question: ")

    if not user_query:
        print("Please enter a question.")
    else:
        try:
            # 1. Get embedding for the user query
            query_embedding = get_embedding(user_query)

            # 2. Search the database for relevant chunks
            search_results = search_database(query_embedding, N_SEARCH_RESULTS)

            # 3. Format the search results as context
            context = format_context(search_results)

            # 4. Generate the answer using the LLM and the context
            if "No relevant information" in context:
                 print(context)
            else:
                # Print the context that will be used by the LLM
                print("\n--- Context Used for Generation ---")
                print(context)
                
                answer = generate_answer(user_query, context)
                print("\n--- Answer ---")
                print(answer)
                # Optionally, print the context that was used
                # print("\n--- Context Used ---")
                # print(context)

        except Exception as e:
            logger.error(f"An error occurred during the query process: {str(e)}")
            print(f"An error occurred: {str(e)}") 