import os
from notion_client import Client
from dotenv import load_dotenv
import sys

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("DATABASE_ID")

notion = Client(auth=NOTION_TOKEN)

def get_transcript_pages():
    response = notion.databases.query(database_id=DATABASE_ID)
    return response["results"]

def get_page_content(page_id):
    blocks = []
    cursor = None
    while True:
        response = notion.blocks.children.list(block_id=page_id, start_cursor=cursor)
        blocks.extend(response["results"])
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return blocks

def extract_text_from_blocks(blocks):
    texts = []
    for block in blocks:
        if block["type"] == "paragraph":
            text_parts = block["paragraph"].get("rich_text", [])
            for part in text_parts:
                texts.append(part["plain_text"])
    return "\n".join(texts)

def extract_all_transcripts():
    pages = get_transcript_pages()
    transcripts = []
    for page in pages:
        # Skip if the page is already indexed
        if page["properties"].get("Indexed", {}).get("checkbox", False):
            continue
            
        title = page["properties"]["Name"]["title"][0]["plain_text"]
        course = page["properties"]["Course"]["select"]["name"]
        page_id = page["id"]

        blocks = get_page_content(page_id)
        transcript_text = extract_text_from_blocks(blocks)

        transcripts.append({
            "page_id": page_id,
            "title": title,
            "course": course,
            "content": transcript_text
        })

    return transcripts

def mark_page_indexed(page_id):
    """Set the 'Indexed' checkbox property on a Notion page to True."""
    notion.pages.update(
        page_id=page_id,
        properties={
            "Indexed": {"checkbox": True}
        }
    )

if __name__ == "__main__":
    transcripts = extract_all_transcripts()
    title_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if title_arg:
        found = False
        for t in transcripts:
            if t["title"] == title_arg:
                print(f"\n--- {t['title']} ({t['course']}) ---\n")
                print(t["content"])
                mark_page_indexed(t["page_id"])
                found = True
                break
        if not found:
            print(f"No transcript found with title: {title_arg}")
    else:
        for t in transcripts:
            print(f"\n--- {t['title']} ({t['course']}) ---\n")
            print(t["content"][:500] + "...\n")
            mark_page_indexed(t["page_id"])