import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import time
import tempfile
import os
# Tool imports
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.python.tool import PythonREPLTool
# Load environment variables
load_dotenv()

# Set up Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create the LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
)

# Set up LangChain memory
memory = ConversationBufferMemory(return_messages=True)

# Set up the conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)

# Add your system prompt as the first message
memory.chat_memory.add_user_message(
    "You are a chill and direct assistant and your name is Shirmeen and you are a student in an age bracket of 20-23. "
    "You’re talking to She types in casual English or Roman Urdu and might make spelling mistakes — that’s okay, just understand her meaning. "
    "Reply in a mix of Urdu and hinglish, but never use Hindi words. If a proper Urdu word isn't available, use English instead. "
    "IT ALWAYS REPLY WITH SOME WORDS OF ENGLISH AND SOME WORDS OF URDU. "
    "Be a little soft but not overly polite — prefers clear, straight-forward replies without emotional or dramatic tone. "
    "Talk casually, like normal friends do. Keep your replies helpful, to the point, and friendly but not fake sweet. "
    "DONT ADD TRANSLATION OF ENGLISH WORDS IN URDU"
)

def main():
    recognizer = sr.Recognizer()
    # Set fixed energy threshold for loud voice
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False

    while True:
        mode = input("Press 1 to speak, 2 to type (or 'exit' to quit): ").strip()
        if mode.lower() in ["exit", "quit"]:
            break

        if mode == "1":
            with sr.Microphone() as source:
                print("Speak now... (Adjusting for noise)")
                recognizer.adjust_for_ambient_noise(source, duration=2)
                
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    
                    # Optional: Save recorded audio for debugging
                    # with open("debug_audio.wav", "wb") as f:
                    #     f.write(audio.get_wav_data())

                    user_input = recognizer.recognize_google(audio, language="en-US")
                    print("You said:", user_input)

                except sr.UnknownValueError:
                    print("Sorry, could not understand the audio.")
                    continue
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    continue
                except sr.WaitTimeoutError:
                    print("You took too long to start speaking.")
                    continue

        elif mode == "2":
            user_input = input("Type your message: ")
        else:
            print("Invalid option. Try again.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        response = conversation.predict(input=user_input)
        print(f"Assistant: {response}")

        # Convert assistant response to audio and play it using pygame
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_audio_path = fp.name
        tts = gTTS(text=response, lang='en', slow=False)
        tts.save(temp_audio_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
        pygame.mixer.music.unload()
        os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
