# Discord Bot with Query Pipeline

A Discord bot that uses a query pipeline to answer questions based on a knowledge base. The bot can be used in a dedicated channel or with a command prefix in any channel. It integrates with Notion to fetch and process information from your Notion workspace.

## Features

- Query processing in dedicated channels or with command prefix
- Rate limiting to prevent abuse
- Configurable command prefix
- Rotating log files
- Error handling and logging
- Environment variable configuration
- Notion integration for knowledge base access
- Semantic search capabilities

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv converto-env
   source converto-env/bin/activate  # On Windows: converto-env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the following variables:
   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token
   DISCORD_CHANNEL_ID=your_dedicated_channel_id
   COMMAND_PREFIX=!ask  # Optional, defaults to !ask
   RATE_LIMIT_WINDOW=60  # Optional, defaults to 60 seconds
   RATE_LIMIT_MAX=5      # Optional, defaults to 5 requests per window
   NOTION_API_KEY=your_notion_api_key  # Required for Notion integration
   NOTION_DATABASE_ID=your_notion_database_id  # Required for Notion integration
   ```

## Usage

1. Start the bot:

   ```bash
   python discord_bot.py
   ```

2. Use the bot in two ways:
   - In the dedicated channel: Simply type your question
   - In other channels: Use the command prefix (default: `!ask`) followed by your question

## Environment Variables

- `DISCORD_BOT_TOKEN`: Your Discord bot token (required)
- `DISCORD_CHANNEL_ID`: ID of the dedicated channel (required)
- `COMMAND_PREFIX`: Command prefix for non-dedicated channels (optional, default: !ask)
- `RATE_LIMIT_WINDOW`: Time window for rate limiting in seconds (optional, default: 60)
- `RATE_LIMIT_MAX`: Maximum requests allowed per window (optional, default: 5)
- `NOTION_API_KEY`: Your Notion API key (required)
- `NOTION_DATABASE_ID`: ID of the Notion database to query (required)

## Notion Integration

The bot uses the Notion API to access and search through your knowledge base. To set this up:

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Get your API key from the integration settings
3. Share your Notion database with the integration
4. Add the database ID to your environment variables

The bot will use semantic search to find relevant information from your Notion database when answering questions.

## Logging

Logs are stored in `discord_bot.log` with rotation (max 5MB per file, keeping 3 backup files).

## Security Considerations

- Never commit your `.env` file
- Keep your Discord bot token secure
- Keep your Notion API key secure
- The bot includes rate limiting to prevent abuse
- Error messages are sanitized to prevent information leakage

## Contributing

Feel free to submit issues and enhancement requests!
