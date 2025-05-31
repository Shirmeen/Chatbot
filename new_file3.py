import os
import requests
import random
# ...existing code...
from datetime import datetime
from gtts import gTTS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper , OpenWeatherMapAPIWrapper
from langchain.agents import Tool
from langchain.chains import LLMMathChain  
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load environment variables FIRST
load_dotenv() 

# Initialize components
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    max_tokens=4000,
    temperature=0.5
)

# Wikipedia tool with parser fix
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        parser_kwargs={"features": "html.parser"}
    )
)

llm_math = LLMMathChain.from_llm(llm=llm)

# Weather tool function
def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = f"https://api.openweathermap.org/data/2.5/weather?id=524901&lang=fr&appid={api_key}"

    try:
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather = {
                "temperature": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "description": data['weather'][0]['description']
            }
            return f"Weather in {city}: {weather['description']}, Temperature: {weather['temperature']}°C, Humidity: {weather['humidity']}%"
        else:
            return f"Error: {data['message']}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
def translate_text(query: str) -> str:
    """Useful for translating text between languages"""
    response = llm.invoke(
        f"Only translate the text to the specified target language. "
        f"Return just the translation without explanations. "
        f"Input: {query}"
    )
    return response.content
# Text-to-Audio tool
def text_to_speech(query: str) -> str:
    """Converts text to audio file"""
    try:
        # Extract text and language from query (format: "text, lang_code")
        if ',' not in query:
            return "Please provide text and language code separated by comma"
            
        text, lang = query.split(',', 1)
        text = text.strip()
        lang = lang.strip().lower()
        
        # Generate unique filename
        filename = f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        
        # Create and save audio
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        
        return f"Audio file generated: {filename}"
    
    except Exception as e:
        return f"Error generating audio: {str(e)}"
def movie_plot_generator(query: str) -> str:
    """Generates hilarious fake movie plots for entertainment"""
    prompt = (
        "Create a ridiculous movie plot combining these elements: "
        f"Genre: {random.choice(['sci-fi', 'rom-com', 'horror', 'documentary'])}, "
        f"Main Character: {random.choice(['a sentient potato', 'a time-traveling barista', 'a vampire accountant'])}. "
        "Include an absurd twist and a punny title. Make it 2-3 sentences max."
    )
    response = llm.invoke(prompt)
    return f"🎬 Feature Presentation:\n{response.content}"
# 1. Configure RAG system
def setup_rag():
    # Load your custom documents (create a 'docs' folder with .txt files)
    file_path = r"C:\Users\shirm\Downloads\project_data_structure_21l-5653_21l-5686.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    loader = TextLoader(file_path) # Add your documents
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

# Initialize RAG system
rag_chain = setup_rag()

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for factual questions about people, places, and events"
    ),
    Tool(
        name="Calculator",
        func=llm_math.run,
        description="Useful for mathematical calculations and arithmetic problems"
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Useful for getting current weather information for a specific city"
    ),
    Tool(
        name="Translator",
        func=translate_text,
        description="Translate text between languages. Examples: "
        "'Translate 'Hello' to French', "
        "'How do you say 'Thank you' in Spanish?'"
    ),
    Tool(
        name="TextToSpeech",
        func=text_to_speech,
        description="Convert text to audio. Input format: 'Text to convert, language_code'. "
                    "Example: 'Hello world, en' for english"
    ),
    Tool(
        name="MoviePlotGenerator",
        func=movie_plot_generator,
        description="Generates absurd movie plots for fun. Example: 'Give me a crazy movie idea'"
    ),
     Tool(
        name="CompanyKnowledgeBase",
        func=lambda query: rag_chain.invoke({"query": query})["result"],
        description="Useful for answering questions about company policies, products, and internal documentation"
    )
    
]


# Set up agent
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat history
chat_history = []

# Start interactive chat loop
print("Welcome to the Chat Assistant! Type 'exit' to end the conversation.")
while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    
    # Process input with chat history
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    # Get agent response
    agent_response = response['output']
    
    # Update chat history
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=agent_response)
    ])
    
    print(f"\nAgent: {agent_response}")