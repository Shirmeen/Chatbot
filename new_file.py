from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Setup LLM (Groq with LLaMA-3)
llm = ChatGroq(model_name="llama-3.1-8b-instant",max_tokens=4000)

# Wikipedia tool setup

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# print(wikipedia.run("HUNTER X HUNTER"))

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for answering general knowledge questions."
    )
]

# Initialize agent (no memory, no speech)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test input loop
import traceback
def truncate_input(text, max_tokens=5500):
    # Simple length-based truncation, not exact token-based
    if len(text) > max_tokens:
        return text[:max_tokens] + "..."
    return text

while True:
    user_input = input("Ask a question (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        break

    try:
        # Truncate user input to stay within safe limits
        

        # Run agent
        response = agent.run(user_input)
        print(f"\nWikipedia Answer:\n{response}\n")

    except Exception as e:
        error_msg = str(e)
        if "token" in error_msg.lower() or "413" in error_msg:
            print("❌ Token limit exceeded. Please shorten your question or input.")
        else:
            print(f"❌ An error occurred:\n{error_msg}")
            traceback.print_exc()

