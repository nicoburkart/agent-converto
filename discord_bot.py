import os
import discord
from discord import app_commands
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import time
from collections import defaultdict
from datetime import datetime, timedelta
import chromadb
from pathlib import Path

# Import functions from your query pipeline
from query_pipeline import get_embedding, search_database, format_context, generate_answer
from prompts import LESSON_SUMMARY_PROMPT

# Import functions from your embed pipeline
from embed_pipeline import extract_all_transcripts, embed_and_store

# Set up logging with rotation
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'discord_bot.log'
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Get Discord bot token and dedicated channel ID from environment variables
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEDICATED_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!ask")  # Configurable command prefix
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Rate limit window in seconds
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "5"))  # Maximum requests per window

# Constants
VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "agent-converto")
CHROMA_DB_PATH = Path("./chroma_db")

# Thread tracking
thread_lessons = {}  # Maps thread_id to (course, lesson) tuple
thread_history = {}  # Maps thread_id to list of (role, content) tuples
thread_content = {}  # Maps thread_id to full lesson content

# Validate required environment variables
required_env_vars = ["DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please add them to your .env file.")
    exit(1)

# Convert channel ID to integer
try:
    DEDICATED_CHANNEL_ID = int(DEDICATED_CHANNEL_ID)
except (ValueError, TypeError):
    logger.error("Invalid DISCORD_CHANNEL_ID in environment variables. Must be an integer.")
    exit(1)

# Rate limiting setup
user_requests = defaultdict(list)

def is_rate_limited(user_id: int) -> bool:
    """Check if a user has exceeded their rate limit."""
    now = datetime.now()
    user_requests[user_id] = [req_time for req_time in user_requests[user_id] 
                            if now - req_time < timedelta(seconds=RATE_LIMIT_WINDOW)]
    
    if len(user_requests[user_id]) >= RATE_LIMIT_MAX:
        return True
    
    user_requests[user_id].append(now)
    return False

# Initialize Discord client with intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def get_all_courses() -> list:
    """Get a list of all unique courses from the ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(VECTOR_DB_COLLECTION)
        
        # Get all metadata
        results = collection.get()
        if not results or not results.get('metadatas'):
            return []
            
        # Extract unique courses from metadata
        courses = set()
        for metadata in results['metadatas']:
            if 'course' in metadata:
                courses.add(metadata['course'])
        
        return sorted(list(courses))
    except Exception as e:
        logger.error(f"Error getting courses: {str(e)}")
        return []

def get_lessons_for_course(course: str) -> list:
    """Get all lessons for a specific course from the ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(VECTOR_DB_COLLECTION)
        
        # Get all metadata
        results = collection.get()
        if not results or not results.get('metadatas'):
            return []
            
        # Filter lessons for the specified course (case-insensitive)
        course_lower = course.lower()
        lessons = []
        for metadata in results['metadatas']:
            if metadata.get('course', '').lower() == course_lower:
                lesson_info = {
                    'title': metadata.get('title', 'Unknown Title'),
                    'course': metadata.get('course', 'Unknown Course')
                }
                if lesson_info not in lessons:  # Avoid duplicates
                    lessons.append(lesson_info)
        
        return sorted(lessons, key=lambda x: x['title'])
    except Exception as e:
        logger.error(f"Error getting lessons for course {course}: {str(e)}")
        return []

def get_all_lessons() -> list:
    """Get a list of all unique lessons from the ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(VECTOR_DB_COLLECTION)
        
        # Get all metadata
        results = collection.get()
        if not results or not results.get('metadatas'):
            return []
            
        # Extract unique lessons from metadata
        lessons = set()
        for metadata in results['metadatas']:
            if 'title' in metadata:
                lessons.add(metadata['title'])
        
        return sorted(list(lessons))
    except Exception as e:
        logger.error(f"Error getting lessons: {str(e)}")
        return []

def get_thread_context(thread_id: int, max_history: int = 5) -> str:
    """Get formatted conversation history for a thread."""
    if thread_id not in thread_history:
        return ""
    
    # Get the last N messages from history
    recent_history = thread_history[thread_id][-max_history:]
    
    # Format the history
    context = "Previous conversation:\n"
    for role, content in recent_history:
        context += f"{role}: {content}\n"
    
    return context

def split_message(content: str, max_length: int = 1900) -> list[str]:
    """Split a message into chunks that fit within Discord's message length limit."""
    if len(content) <= max_length:
        return [content]
    
    chunks = []
    current_chunk = ""
    
    # Split by newlines first to keep paragraphs together
    paragraphs = content.split('\n')
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, start a new chunk
        if len(current_chunk) + len(paragraph) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += '\n'
            current_chunk += paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

async def send_long_message(channel, content: str):
    """Send a message that might be longer than Discord's limit by splitting it into multiple messages."""
    chunks = split_message(content)
    for chunk in chunks:
        await channel.send(chunk)

@tree.command(name="courses", description="List all available courses")
async def courses_command(interaction: discord.Interaction):
    """Slash command to list all available courses."""
    await interaction.response.defer()
    
    try:
        courses = get_all_courses()
        if not courses:
            await interaction.followup.send("No courses found in the database.")
            return
            
        # Create a formatted message
        message = "**Available Courses:**\n"
        for i, course in enumerate(courses, 1):
            message += f"{i}. {course}\n"
            
        await interaction.followup.send(message)
    except Exception as e:
        logger.error(f"Error in courses command: {str(e)}")
        await interaction.followup.send("An error occurred while fetching courses.")

@tree.command(name="sync_notion", description="Index newly added Notion transcripts into the knowledge base")
async def sync_notion_command(interaction: discord.Interaction):
    """Slash command to trigger Notion indexing."""
    await interaction.response.defer(ephemeral=True) # Use ephemeral=True so only the user sees the command initiation

    notion_token = os.getenv("NOTION_TOKEN")
    database_id = os.getenv("DATABASE_ID")

    if not notion_token or not database_id:
        await interaction.followup.send("Error: Notion token or Database ID is not set in environment variables. Cannot sync.")
        return

    try:
        await interaction.followup.send("Starting Notion indexing...")
        logger.info("Starting Notion indexing triggered by /sync_notion")

        transcripts = extract_all_transcripts()

        if transcripts:
            await interaction.followup.send(f"Found {len(transcripts)} new transcripts. Embedding and storing...")
            embed_and_store(transcripts)
            await interaction.followup.send("Notion indexing complete!")
            logger.info("Notion indexing completed successfully")
        else:
            await interaction.followup.send("No new transcripts found to index.")
            logger.info("No new transcripts found during /sync_notion")

    except Exception as e:
        logger.error(f"Error during Notion sync: {str(e)}", exc_info=True)
        await interaction.followup.send(f"An error occurred during Notion sync: {str(e)}")

@tree.command(name="lessons", description="List all lessons for a specific course")
async def lessons_command(interaction: discord.Interaction, course: str):
    """Slash command to list all lessons for a specific course."""
    await interaction.response.defer()
    
    try:
        lessons = get_lessons_for_course(course)
        if not lessons:
            await interaction.followup.send(f"No lessons found for course: {course}")
            return
            
        # Create a formatted message
        message = f"**Lessons for {course}:**\n"
        for i, lesson in enumerate(lessons, 1):
            message += f"{i}. {lesson['title']}\n"
            
        await interaction.followup.send(message)
    except Exception as e:
        logger.error(f"Error in lessons command: {str(e)}")
        await interaction.followup.send("An error occurred while fetching lessons.")

@lessons_command.autocomplete('course')
async def lessons_course_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete function for course names in the lessons command."""
    courses = get_all_courses()
    return [
        app_commands.Choice(name=course, value=course)
        for course in courses
        if current.lower() in course.lower()
    ][:25]  # Discord limits autocomplete to 25 choices

@tree.command(name="summary", description="Get a summary of a specific lesson from a course")
async def summary_command(interaction: discord.Interaction, course: str, lesson: str):
    """Slash command to get a summary of a specific lesson from a course."""
    await interaction.response.defer()
    
    try:
        # Get the lesson content from ChromaDB
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(VECTOR_DB_COLLECTION)
        
        # Search for the lesson by title and course using $and operator
        results = collection.get(
            where={
                "$and": [
                    {"title": lesson},
                    {"course": course}
                ]
            }
        )
        
        if not results or not results.get('documents'):
            await interaction.followup.send(f"No content found for lesson '{lesson}' in course '{course}'")
            return
            
        # Get the lesson content
        lesson_content = results['documents'][0]
        
        # Generate a summary directly from the lesson content
        summary = generate_answer(LESSON_SUMMARY_PROMPT, lesson_content)
        
        # Create a thread for the discussion
        thread = await interaction.channel.create_thread(
            name=f"Discussion: {lesson} ({course})",
            auto_archive_duration=1440  # 24 hours
        )
        
        # Store the lesson context in our thread tracking
        thread_lessons[thread.id] = (course, lesson)
        thread_history[thread.id] = []  # Initialize empty history
        thread_content[thread.id] = lesson_content
        
        # Send the summary in the thread
        await thread.send(f"**Summary of {lesson} from {course}:**\n{summary}\n\nYou can ask follow-up questions about this lesson in this thread!")
        
        # Add the summary to thread history
        thread_history[thread.id].append(("Assistant", summary))
        
        # Send a confirmation in the original channel with a link to the thread
        await interaction.followup.send(f"Created a thread for discussing {lesson} from {course}! Jump in: {thread.mention}")
        
    except Exception as e:
        logger.error(f"Error in summary command: {str(e)}")
        await interaction.followup.send("An error occurred while generating the summary.")

@summary_command.autocomplete('course')
async def course_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete function for course names."""
    courses = get_all_courses()
    return [
        app_commands.Choice(name=course, value=course)
        for course in courses
        if current.lower() in course.lower()
    ][:25]  # Discord limits autocomplete to 25 choices

@summary_command.autocomplete('lesson')
async def lesson_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete function for lesson names."""
    # Get the selected course from the interaction
    course = interaction.data.get('options', [])[0].get('value')
    if not course:
        return []
        
    # Get lessons for the selected course
    lessons = get_lessons_for_course(course)
    return [
        app_commands.Choice(name=lesson['title'], value=lesson['title'])
        for lesson in lessons
        if current.lower() in lesson['title'].lower()
    ][:25]  # Discord limits autocomplete to 25 choices

@client.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'Logged in as {client.user.name}')
    print(f'Bot is ready. Logged in as {client.user.name}')
    
    # Sync the command tree with Discord
    await tree.sync()
    logger.info("Slash commands synced with Discord")

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    query = None
    is_thread_message = False
    thread_context = None

    # Check if the message is in a thread
    if isinstance(message.channel, discord.Thread):
        is_thread_message = True
        thread_context = thread_lessons.get(message.channel.id)
        if thread_context:
            query = message.content.strip()
            logger.info(f"Received follow-up question in thread {message.channel.name}: {query}")
    # Check if the message is in the dedicated channel
    elif message.channel.id == DEDICATED_CHANNEL_ID:
        query = message.content.strip()
        logger.info(f"Received direct query in dedicated channel {message.channel.name}: {query}")
    elif message.content.startswith(COMMAND_PREFIX):
        query = message.content[len(COMMAND_PREFIX):].strip()
        logger.info(f"Received {COMMAND_PREFIX} query in channel {message.channel.name}: {query}")

    # Process the query if one was identified
    if query:
        if not query:
            response_text = "Please provide a question."
            if not is_thread_message and message.channel.id != DEDICATED_CHANNEL_ID:
                response_text = f"Please provide a question after the {COMMAND_PREFIX} command."
            await message.channel.send(response_text)
            return

        # Check rate limit
        if is_rate_limited(message.author.id):
            await message.channel.send(f"Rate limit exceeded. Please wait {RATE_LIMIT_WINDOW} seconds between requests.")
            return

        # Indicate that the bot is processing the request
        async with message.channel.typing():
            try:
                # If this is a thread message, add the lesson context to the query
                if is_thread_message and thread_context:
                    course, lesson = thread_context
                    context_query = f"Regarding {lesson} from {course}: {query}"
                    
                    # Add user's question to thread history
                    thread_history[message.channel.id].append(("User", query))
                    
                    # Get conversation history
                    conversation_context = get_thread_context(message.channel.id)
                    
                    # Get the full lesson content for this thread
                    full_lesson_content = thread_content.get(message.channel.id, "")
                    
                    # Use your existing query pipeline functions to get related content from other lessons
                    query_embedding = get_embedding(context_query)
                    search_results = search_database(query_embedding)
                    related_context = format_context(search_results)
                    
                    # Combine all context: conversation history, full lesson content, and related content
                    context = f"{conversation_context}\n\n"
                    if full_lesson_content:
                        context += f"Full lesson content:\n{full_lesson_content}\n\n"
                    if related_context and "No relevant information" not in related_context:
                        context += f"Related content from other lessons:\n{related_context}"
                else:
                    context_query = query
                    conversation_context = ""
                    # Use your existing query pipeline functions
                    query_embedding = get_embedding(context_query)
                    search_results = search_database(query_embedding)
                    context = format_context(search_results)
                    
                    # Combine the conversation history with the search results
                    if conversation_context:
                        context = f"{conversation_context}\n\n{context}"

                if "No relevant information" in context:
                    response = context
                else:
                    response = generate_answer(context_query, context)

                # Send the answer back to the Discord channel
                await send_long_message(message.channel, response)
                
                # Add bot's response to thread history
                if is_thread_message:
                    thread_history[message.channel.id].append(("Assistant", response))

            except Exception as e:
                logger.error(f"An error occurred during query processing: {str(e)}", exc_info=True)
                await message.channel.send("An error occurred while processing your request. Please try again later.")

# Add thread cleanup when it's archived
@client.event
async def on_thread_update(before, after):
    """Clean up thread data when a thread is archived."""
    if after.archived and before.id in thread_lessons:
        del thread_lessons[before.id]
        if before.id in thread_history:
            del thread_history[before.id]
        if before.id in thread_content:
            del thread_content[before.id]
        logger.info(f"Cleaned up thread data for archived thread: {before.name}")

# Run the Discord bot
if __name__ == "__main__":
    try:
        client.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}", exc_info=True) 