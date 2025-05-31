from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set up LangChain
llm = ChatGroq(model_name="llama-3.1-8b-instant")
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

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

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"response": "No input provided."}), 400
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})

# Optional: Add audio endpoint if you want to process audio
@app.route("/chat/audio", methods=["POST"])
def chat_audio():
    # This is a placeholder. You need to implement speech-to-text here.
    return jsonify({"response": "Audio processing not implemented."}), 501

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)