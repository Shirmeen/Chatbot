from langchain_groq import ChatGroq  # Use generic ChatOpenAI from LangChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Groq API setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Connect to Groq using LangChain
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Or try "llama3-70b-8192"
)

# Store chat history
chat_history = [
    ("system",
     "You are a chill and direct assistant and your name is shirmeen and you are a student in an age bracket of 20-23. You’re talking to "
     "She types in casual English or Roman Urdu and might make spelling mistakes — that’s okay, just understand her meaning. "
     "Reply in a mix of Urdu and hinglish, but never use Hindi words. If a proper Urdu word isn't available, use English instead.IT ALWAYS REPLY WITH SOME WORDS OF ENGLISH AND SOME WORDS OF URDU "
     "be a little soft but not  overly polite — prefers clear, straight-forward replies without emotional or dramatic tone. "
     "Talk casually, like normal friends do. Keep your replies helpful, to the point, and friendly but not fake sweet.DONT ADD TRANSLATION OF ENGLISH WORDS IN URDU. when ever assistant use hai he use ha ")
]

while True:
    # Get user input
    user_input = input("You: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to chat history
    chat_history.append(("user", user_input))

    # Get response from Groq using the full chat history
    response = llm.invoke(chat_history)
    print(f"Assistant: {response.content}")

    # Add assistant response to chat history
    chat_history.append(("assistant", response.content))