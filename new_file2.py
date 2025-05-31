import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import pyttsx3
import os

# ✅ Wikipedia tool setup
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Setup LLM with Groq and LLaMA-3
llm = ChatGroq(model_name="llama-3.1-8b-instant")

# Memory setup
memory = ConversationBufferMemory(return_messages=True)

# Wikipedia tool
wiki = WikipediaAPIWrapper()
tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for answering general knowledge questions."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

# Add system prompt
memory.chat_memory.add_user_message(
    "You are a chill and direct assistant and your name is Shirmeen and you are a student in an age bracket of 20-23. "
    "You’re talking to She types in casual English or Roman Urdu and might make spelling mistakes — that’s okay, just understand her meaning. "
    "Reply in a mix of Urdu and hinglish, but never use Hindi words. If a proper Urdu word isn't available, use English instead. "
    "IT ALWAYS REPLY WITH SOME WORDS OF ENGLISH AND SOME WORDS OF URDU. "
    "Be a little soft but not overly polite — prefers clear, straight-forward replies without emotional or dramatic tone. "
    "Talk casually, like normal friends do. Keep your replies helpful, to the point, and friendly but not fake sweet. "
    "DONT ADD TRANSLATION OF ENGLISH WORDS IN URDU"
)

# Speak response (faster)
def speak_fast(text, rate=300):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Main loop
def main():
    recognizer = sr.Recognizer()

    while True:
        mode = input("Press 1 to speak, 2 to type (or 'exit' to quit): ").strip()
        if mode.lower() in ["exit", "quit"]:
            break

        if mode == "1":
            with sr.Microphone() as source:
                print("Speak now...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, phrase_time_limit=7)
                try:
                    user_input = recognizer.recognize_google(audio, language="en-US")
                    print("You said:", user_input)
                except sr.UnknownValueError:
                    print("Sorry, could not understand the audio.")
                    continue
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    continue

        elif mode == "2":
            user_input = input("Type your message: ")
        else:
            print("Invalid option. Try again.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        # ✅ Replaced deprecated run() with invoke()
        response = agent.invoke(user_input)
        print(f"Assistant: {response}")

        speak_fast(response, rate=300)

if __name__ == "__main__":
    main()
