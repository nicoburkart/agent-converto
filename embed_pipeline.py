import os
from dotenv import load_dotenv
from notion.extract import extract_all_transcripts, mark_page_indexed
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import chromadb
from pathlib import Path
import logging
from typing import List, Dict, Any
import openai
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "agent-converto")
OPENAI_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DB_PATH = Path("./chroma_db")
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds between requests
BATCH_SIZE = 5  # number of texts to process in one batch

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    # Decide if you want to exit or continue based on context (e.g., just checking DB might not need API key)
    # For now, we'll raise the error only if not checking the database
    # This will be handled in the main block after parsing arguments
    # raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

def chunk_transcript(text: str, model_name: str = OPENAI_MODEL, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    try:
        enc = tiktoken.encoding_for_model(model_name)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda txt: len(enc.encode(txt)),
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error chunking transcript: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI API to get embeddings for a list of texts with rate limiting and retries.
    """
    try:
        logger.info(f"Requesting embeddings for {len(texts)} texts...")
        response = openai.embeddings.create(
            model=OPENAI_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except openai.RateLimitError as e:
        logger.warning(f"Rate limit hit, waiting {RATE_LIMIT_DELAY} seconds...")
        time.sleep(RATE_LIMIT_DELAY)
        raise
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

def process_in_batches(texts: List[str]) -> List[List[float]]:
    """
    Process texts in smaller batches to avoid rate limits.
    """
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            batch_embeddings = embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            if i + BATCH_SIZE < len(texts):  # Don't sleep after the last batch
                time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
            raise
    return all_embeddings

def embed_and_store(transcripts: List[Dict[str, Any]]) -> None:
    try:
        logger.info(f"Initializing vector store collection '{VECTOR_DB_COLLECTION}'")
        # Initialize ChromaDB with persistence
        CHROMA_DB_PATH.mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_or_create_collection(VECTOR_DB_COLLECTION)

        for page in transcripts:
            try:
                logger.info(f"Processing page {page['page_id']}: '{page['title']}' (Course: {page['course']})")
                chunks = chunk_transcript(page["content"])
                logger.info(f"Chunked into {len(chunks)} segments")
                
                vectors = process_in_batches(chunks)
                
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    ids.append(f"{page['page_id']}_{i}")
                    embeddings.append(vec)
                    metadatas.append({
                        "page_id": page["page_id"],
                        "title": page["title"],
                        "course": page["course"],
                        "chunk_index": i
                    })
                    documents.append(chunk)
                    
                collection.upsert(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Upserted {len(ids)} documents for page {page['page_id']}")
                
                # Mark the page as indexed in Notion after successful storage
                try:
                    mark_page_indexed(page["page_id"])
                    logger.info(f"Successfully marked page {page['page_id']} as indexed in Notion")
                except Exception as e:
                    logger.error(f"Error marking page {page['page_id']} as indexed in Notion: {str(e)}")
                    # Decide if you want to re-raise or just log and continue
                    pass # continue to the next page

            except Exception as e:
                logger.error(f"Error processing page {page.get('page_id', 'unknown')}: {str(e)}")
                continue

        logger.info(f"Completed embedding and storing for {len(transcripts)} pages.")
    except Exception as e:
        logger.error(f"Error in embed_and_store: {str(e)}")
        raise

def check_database_contents(limit: int = 5) -> None:
    """
    Connects to the ChromaDB and prints the contents.

    Args:
        limit: The maximum number of items to display.
    """
    logger.info(f"Connecting to ChromaDB collection '{VECTOR_DB_COLLECTION}' at '{CHROMA_DB_PATH}'")
    try:
        # Initialize ChromaDB with persistence
        if not CHROMA_DB_PATH.exists():
             logger.warning(f"ChromaDB path does not exist: {CHROMA_DB_PATH}. No database to check.")
             return
             
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

        # Get the collection
        try:
            collection = client.get_collection(VECTOR_DB_COLLECTION)

            # Get the count of items in the collection
            count = collection.count()
            logger.info(f"Total items in collection '{VECTOR_DB_COLLECTION}': {count}")

            if count > 0:
                # Use `limit` to control how many items to retrieve
                results = collection.peek(limit=limit)
                
                # Safely extract ids and metadatas based on the observed structure (list of dicts)
                ids_list = results.get('ids', [])
                metadatas_list = results.get('metadatas', [])

                if ids_list and metadatas_list and len(ids_list) == len(metadatas_list):
                    print(f"\nFirst {len(ids_list)} items (ID, Course, Title, Chunk Index):") # Print actual number of results
                    for i in range(len(ids_list)):
                        item_id = ids_list[i]
                        metadata = metadatas_list[i] # Direct access to the metadata dictionary
                        title = metadata.get('title', 'N/A')
                        course = metadata.get('course', 'N/A')
                        chunk_index = metadata.get('chunk_index', 'N/A')
                        print(f"ID: {item_id}, Course: {course}, Title: {title}, Chunk Index: {chunk_index}")
                else:
                    logger.warning("Could not extract items from peek results. IDs or metadatas lists might be missing, empty, or misaligned.")

            else:
                logger.info("No items in the collection yet.")

        except Exception as e:
            logger.error(f"Error accessing ChromaDB collection '{VECTOR_DB_COLLECTION}': {e}")

    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Notion transcripts and store in ChromaDB.")
    parser.add_argument(
        '--check-db', 
        action='store_true', 
        help='Check the contents of the ChromaDB instead of running the embedding pipeline.'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=5, # Default limit is 5
        help='Number of items to display when using --check-db.'
    )

    args = parser.parse_args()

    if args.check_db:
        check_database_contents(limit=args.limit)
    else:
        # Validate required environment variables only if running the embedding pipeline
        required_env_vars = ["OPENAI_API_KEY", "NOTION_TOKEN", "DATABASE_ID"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

        try:
            logger.info("🔄 Extracting all transcripts from Notion...")
            transcripts = extract_all_transcripts()
            logger.info(f"✅ Extracted {len(transcripts)} transcripts.")
            if transcripts:
                embed_and_store(transcripts)
            else:
                logger.info("No new transcripts to process.")
                
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            raise