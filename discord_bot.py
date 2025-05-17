import os
import discord
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Import functions from your query pipeline
from query_pipeline import get_embedding, search_database, format_context, generate_answer

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

# Define the Discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user.name}')
    print(f'Bot is ready. Logged in as {client.user.name}')

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    query = None

    # Check if the message is in the dedicated channel
    if message.channel.id == DEDICATED_CHANNEL_ID:
        query = message.content.strip()
        logger.info(f"Received direct query in dedicated channel {message.channel.name}: {query}")
    elif message.content.startswith(COMMAND_PREFIX):
        query = message.content[len(COMMAND_PREFIX):].strip()
        logger.info(f"Received {COMMAND_PREFIX} query in channel {message.channel.name}: {query}")

    # Process the query if one was identified
    if query:
        if not query:
            response_text = "Please provide a question."
            if message.channel.id != DEDICATED_CHANNEL_ID:
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
                # Use your existing query pipeline functions
                query_embedding = get_embedding(query)
                search_results = search_database(query_embedding)
                context = format_context(search_results)

                if "No relevant information" in context:
                    response = context
                else:
                    response = generate_answer(query, context)

                # Send the answer back to the Discord channel
                await message.channel.send(response)

            except Exception as e:
                logger.error(f"An error occurred during query processing: {str(e)}", exc_info=True)
                await message.channel.send("An error occurred while processing your request. Please try again later.")

# Run the Discord bot
if __name__ == "__main__":
    try:
        client.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}", exc_info=True) 