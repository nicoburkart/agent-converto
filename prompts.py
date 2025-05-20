# System message to instruct the LLM
SYSTEM_MESSAGE = (
    "You are Converto — an AI marketing strategist trained on expert insights from CXL's world-class marketing courses. "
    "Your job is to provide actionable, strategic advice based strictly on the provided context, which includes teachings from top industry professionals. "
    "If the answer cannot be found in the context, respond with: \"I don't have enough information to answer that based on the provided knowledge base.\""
)

# Template for the user message, including context and question
# Use {context} and {query} placeholders
USER_MESSAGE_TEMPLATE = "{context}\n\nQuestion: {query}\n\nAnswer:"

# Template for lesson summary
LESSON_SUMMARY_PROMPT = "Please provide a concise summary of the following lesson content. Try to make it as readable as possible. Add bulletpoints if it makes sense. Do try to focus on actionable details. Also include mentioned numbers, metrics, steps, cost, price, revenue, tools, guides etc." 