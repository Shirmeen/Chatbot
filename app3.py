from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import ast
import re
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# 1. Load environment variables from your .env file
# This must be called at the very beginning to load GOOGLE_API_KEY
load_dotenv()

# --- Configuration for the Gemini LLM ---
# Get the Google API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini LLM using ChatGoogleGenerativeAI
# We're using 'gemini-1.5-flash-latest' as confirmed to be working.
# 'temperature' controls creativity (0.3 is good for focused tasks).
# 'convert_system_message_to_human=True' is important for LangChain's Gemini integration.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=google_api_key,
    temperature=0.3,
    convert_system_message_to_human=True
)
# Non-technical description provided by the client
non_technical_description = (
    "I want to build a chatbot that answers customer questions using information from our product manuals. "
    "It should sound smart and respond fast, even when lots of people ask at once."
)

# Developer skills list
developer_skills = [
    "Python", "React", "AWS", "LangChain", "Docker", "PyTorch", "TensorFlow", "FastAPI", "Flask",
    "PostgreSQL", "MongoDB", "JavaScript", "TypeScript", "Node.js", "Git", "Linux",
    "Kubernetes", "Redis", "GraphQL", "HTML", "CSS", "machine learning"
]

# Agent 1: Requirement Analyzer 🕵️
requirement_analyzer = Agent(
    role='Requirement Analyzer 🕵️',
    goal='Analyze non-technical client input and extract the technical skills required for the task',
    backstory="""Specializes in interpreting non-technical project descriptions from clients and identifying the technical tools, libraries, or frameworks needed to implement it.""",
    llm=llm,
    verbose=True
)
# Agent 2: Skill Comparer 🤹
skill_comparer = Agent(
    role='Skill Comparer 🤹',
    goal=(
        "Compare developer skills with required client skills. "
        "Categorize the skills into: exact matches and missing skills. Also very important same skills are added in the matched skills. "
        "For example, if the developer knows Natural Language Processing, it should be considered a similar skill to Machine Learning."
        "Provides a breakdown into two categories:\n"
        "- matched_skills (exact matches or e.g. Natural Language Processing ≈ Machine Learning, Flask ≈ Django)\n"
        "- missing_skills (no match or related skill found IN the developer's list)"
        
    ),
    backstory=(
        "Takes two lists: required client skills and developer skills. "

        "Identifies direct matches, then uses common industry knowledge to detect closely related or complementary skills. "
    ),
    llm=llm,
    verbose=True
)
# Agent 3: Relation Mapper 🧑‍🏫
relation_mapper = Agent(
    role='Skill Learning Difficulty Assessor 🧑‍🏫',
    goal=(
        "Given a list of the developer's skills and the missing skills, "
        "analyze for each missing skill how easily a developer with these skills could learn it. "
        "Consider overlap, prerequisites, and domain similarity. "
        "Return a Python dict mapping each missing skill to 'Easy', 'Moderate', or 'Difficult' to learn."
    ),
    backstory=(
        "You are an expert in developer education. For each missing skill, "
        "assess the learning curve based on the developer's existing skills. "
        "If a missing skill is a framework or library for a language the developer knows, mark as 'Easy'. "
        "If it's in a related domain, mark as 'Moderate'. Otherwise, mark as 'Difficult'. "
    ),
    llm=llm,
    verbose=True
)

# Agent 4: Confidence Scorer 🏆
confidence_scorer = Agent(
    role='Confidence Scorer 🏆',
    goal=(
        "Given the matched skills (including similar skills) and the learning difficulty assessment for missing skills, "
        "calculate an overall confidence score (0-100%) for the developer's ability to fulfill the client requirements. "
        "If all or most missing skills are 'Easy', confidence should be very high (90-100). "
        "If most are 'Easy' or 'Moderate', confidence should be high (70-89). "
        "If most are 'Moderate', confidence should be moderate (50-69). "
        "If there are any 'Difficult', reduce confidence accordingly (below 50). "
        "Return ONLY an integer percentage, no explanation, no thoughts, no text, just the number."
    ),
    backstory=(
        "You are an expert at evaluating project fit. "
        "You use both the overlap of skills and the learning curve for missing skills to estimate how likely the developer is to succeed. "
        "You must return ONLY an integer percentage, nothing else."
    ),
    llm=llm,
    verbose=True
)
# Task 1: Requirement Analysis Task
task1 = Task(
    description=(
        f"""Extract the technical skills, libraries, frameworks, or tools 
        from a non-technical project description. ONLY return a Python list of strings.
        No explanations. No thoughts. No extra text.

        Respond exactly like this:
        ['LangChain', 'Flask', 'Kubernetes', 'GPT-4']

        Here is the client request:
        \"{non_technical_description}\""""
    ),
    expected_output="A valid Python list of strings like ['LangChain', 'Flask']. No explanations or extra text.",
    agent=requirement_analyzer
)

# Task 2: Skill Matching Task
task2 = Task(
description=(
    f"""You are given:
1. A list of developer's skills:\n{developer_skills}
2. A list of required client skills (from Task 1).

Instructions:
- Compare both lists and return:
  a. 'matched_skills': Skills that are either exactly the same  includes similar matches OR closely related (e.g., Natural Language processing = Machine Learning, Flask = Django).
  b. 'missing_skills': Skills that do not exist in the developer's list and are not closely related to developer's skills.

Return ONLY a Python dictionary in this exact format:
{{
  'matched_skills': [...],  # includes exact and similar matches
  'missing_skills': [...]  
}}

No explanation. No extra text."""
)
,
    expected_output="A Python dict with keys 'matched_skills' and 'missing_skills'.",
    agent=skill_comparer
)


# Task 3: Project Planning Task
task3 = Task(
            description=(
                f"""Given the developer's skills:\n{developer_skills}\n
                And the missing skills extracted from Task 2.\n

                For each missing skill, analyze how easy or difficult it would be for the developer to learn it,
                based on their existing skills. Consider overlap, prerequisites, and domain similarity.

                Return a Python dict mapping each missing skill to one of: 'Easy', 'Moderate', or 'Difficult'.

                Example:
                {{
                    'Kubernetes': 'Easy',
                    'GPT-4': 'Moderate',
                    'Flutter': 'Difficult'(Skills which are unable to understand really easily)
                }}

                Do not explain. Just return the Python dict."""
            ),
            expected_output="A Python dict mapping each missing skill to 'Easy', 'Moderate', or 'Difficult'.",
            agent=relation_mapper

)
task4 = Task(
            description=(
                f"""You are given:
               Matched skills  extracted from Task 2.\n
Learning difficulty assessment  extracted from Task 3.\n

Instructions:
1. Count the total number of required skills (matched + missing).
2. Calculate the percentage of matched skills: (number of matched skills / total required skills) * 100.
3. Calculate the percentage of missing skills that are 'Easy' to learn: (number of 'Easy' / total required skills) * 100. Add this to the score.
4. Calculate the percentage of missing skills that are 'Difficult' to learn: (number of 'Difficult' / total required skills) * 100. Subtract this from the score.
5. The final confidence score is: matched percentage + easy percentage - difficult percentage.
6. Return ONLY the final integer percentage. No explanation, no thoughts, no text, just the number.
"""
            ),
            expected_output="An integer percentage  (1- 100) representing confidence.",
            agent=confidence_scorer
        )

# Create the crew with all agents and tasks
crew = Crew(
    agents=[requirement_analyzer, skill_comparer, relation_mapper, confidence_scorer],
    tasks=[task1, task2, task3,task4],
    verbose=True
)

# Run the crew
if __name__ == "__main__":
    print("\n🔍 Starting Skill Extraction and Matching Process...\n")
    result = crew.kickoff()
    print("\n✅ Skill Matching Results:\n")
    print(result)
