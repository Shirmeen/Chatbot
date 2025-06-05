import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import logging # Import the logging module

# 1. Load environment variables from your .env file
load_dotenv()

logging.getLogger("litellm").setLevel(logging.WARNING)

# --- Configuration for the Gemini LLM ---
# Get the Google API key from environment variables
google_api_key = os.getenv("Groq_API_KEY")

# Initialize the Gemini LLM using ChatGoogleGenerativeAI
llm = ChatGroq(
    model="gemini-1.5-flash-latest",
    api_key=google_api_key,
    temperature=0.3,
    convert_system_message_to_human=True
)

# --- Define Project Inputs ---
non_technical_description = (
    "I want to build a chatbot that answers customer questions using information from our product manuals. "
    "It should sound smart and respond fast, even when lots of people ask at once."
)

developer_skills = [
    "Python", "React", "AWS", "LangChain", "Docker", "PyTorch", "TensorFlow", "FastAPI", "Flask",
    "PostgreSQL", "MongoDB", "JavaScript", "TypeScript", "Node.js", "Git", "Linux",
    "Kubernetes", "Redis", "GraphQL", "HTML", "CSS", "machine learning"
]

# --- Define Agents ---
requirement_analyzer = Agent(
    role='Requirement Analyzer 🕵️',
    goal='Analyze non-technical client input and extract the technical skills required for the task',
    backstory="""Specializes in interpreting non-technical project descriptions from clients and
                 identifying the technical tools, libraries, or frameworks needed to implement it.
                 Always returns a Python list of strings.""",
    llm=llm,
    verbose=True
)

skill_comparer = Agent(
    role='Skill Comparer 🤹',
    goal=(
        "Compare developer skills with required client skills. "
        "Categorize the skills into: exact matches and missing skills. Very important: similar skills "
        "should be added to matched_skills (e.g., Natural Language Processing ≈ Machine Learning, Flask ≈ Django). "
        "Return a breakdown into two categories:\n"
        "- 'matched_skills' (exact matches or similar/complementary skills)\n"
        "- 'missing_skills' (no direct or related match found in the developer's list)"
    ),
    backstory=(
        "You take two lists: required client skills and developer skills. "
        "You identify direct matches, then use common industry knowledge to detect closely related "
        "or complementary skills (e.g., Flask is similar to Django, both being web frameworks). "
        "Always return a Python dictionary with 'matched_skills' and 'missing_skills' keys."
    ),
    llm=llm,
    verbose=True
)

learning_difficulty_assessor = Agent(
    role='Skill Learning Difficulty Assessor 🧑‍�',
    goal=(
        "Given a list of the developer's existing skills and the missing skills, "
        "analyze for each missing skill how easily a developer with these existing skills could learn it. "
        "Consider overlap, prerequisites, and domain similarity. "
        "Return a Python dict mapping each missing skill to 'Easy', 'Moderate', or 'Difficult' to learn."
    ),
    backstory=(
        "You are an expert in developer education. For each missing skill, "
        "assess the learning curve based on the developer's existing skills. "
        "If a missing skill is a framework or library for a language the developer knows, mark as 'Easy'. "
        "If it's in a related domain, mark as 'Moderate'. Otherwise, mark as 'Difficult'. "
        "Always return a Python dictionary."
    ),
    llm=llm,
    verbose=True
)

confidence_scorer = Agent(
    role='Confidence Scorer 🏆',
    goal=(
        "Given the matched skills (including similar skills) and the learning difficulty assessment for missing skills, "
        "calculate an overall confidence score (0-100%) for the developer's ability to fulfill the client requirements. "
        "If all or most missing skills are 'Easy', confidence should be very high (90-100). "
        "If most are 'Easy' or 'Moderate', confidence should be high (70-89). "
        "If most are 'Moderate', confidence should be moderate (50-69). "
        "If there are any 'Difficult' skills, reduce confidence accordingly (below 50). "
        "Return ONLY an integer percentage (e.g., 85), no explanation, no thoughts, no extra text, just the number."
    ),
    backstory=(
        "You are an expert at evaluating project fit. "
        "You use both the overlap of skills and the learning curve for missing skills to estimate how likely the developer is to succeed. "
        "You must return ONLY an integer percentage, nothing else."
    ),
    llm=llm,
    verbose=True
)

# --- Define Tasks ---
task1_analyze_requirements = Task(
    description=(
        f"""Extract the technical skills, libraries, frameworks, or tools
        from the following non-technical project description.
        ONLY return a Python list of strings. No explanations. No thoughts. No extra text.

        Respond exactly like this:
        ['LangChain', 'Flask', 'Kubernetes', 'GPT-4']

        Client Request: \"{non_technical_description}\""""
    ),
    expected_output="A valid Python list of strings, e.g., ['LangChain', 'Flask']. No explanations or extra text.",
    agent=requirement_analyzer
)

task2_compare_skills = Task(
    description=(
        f"""You are given:
1. A list of developer's skills: {developer_skills}
2. A list of required client skills (from the previous task's output).

Instructions:
- Compare both lists.
- Return 'matched_skills': Skills that are either exactly the same OR closely related (e.g., Natural Language Processing ≈ Machine Learning, Flask ≈ Django).
- Return 'missing_skills': Skills that do not exist in the developer's list and are not closely related to developer's skills.

Return ONLY a Python dictionary in this exact format:
{{
    'matched_skills': [...],
    'missing_skills': [...]
}}
No explanation. No extra text.
"""
    ),
    expected_output="A Python dictionary with keys 'matched_skills' and 'missing_skills'.",
    agent=skill_comparer,
    context=[task1_analyze_requirements]
)

task3_assess_learning_difficulty = Task(
    description=(
        f"""Given the developer's skills: {developer_skills}
        And the missing skills extracted from the previous task's output.

        For each missing skill, analyze how easy or difficult it would be for the developer to learn it,
        based on their existing skills. Consider overlap, prerequisites, and domain similarity.

        Return a Python dict mapping each missing skill to one of: 'Easy', 'Moderate', or 'Difficult'.

        Example:
        {{
            'Kubernetes': 'Easy',
            'GPT-4': 'Moderate',
            'Flutter': 'Difficult'
        }}
        Do not explain. Just return the Python dict."""
    ),
    expected_output="A Python dictionary mapping each missing skill to 'Easy', 'Moderate', or 'Difficult'.",
    agent=learning_difficulty_assessor,
    context=[task2_compare_skills]
)

task4_calculate_confidence = Task(
    description=(
        f"""You are given:
        Matched skills (including similar skills) from the output of Task 2.
        Learning difficulty assessment from the output of Task 3.

        Instructions:
        1. Get the list of matched skills from Task 2's output.
        2. Get the dictionary of missing skills and their learning difficulty from Task 3's output.
        3. Determine the total number of required skills by summing the count of matched skills and missing skills.
        4. Calculate the percentage of matched skills: (number of matched skills / total required skills) * 100.
        5. For missing skills:
           - Add 10 points for each 'Easy' skill.
           - Subtract 10 points for each 'Difficult' skill. (Adjust points as needed for balance)
        6. The final confidence score is: (matched percentage) + (sum of easy/difficult points).
           Ensure the score is between 0 and 100.
        7. Return ONLY the final integer percentage. No explanation, no thoughts, no text, just the number.
        """
    ),
    expected_output="An integer percentage (0-100) representing confidence.",
    agent=confidence_scorer,
    context=[task2_compare_skills, task3_assess_learning_difficulty]
)

# --- Create the Crew ---
crew = Crew(
    agents=[requirement_analyzer, skill_comparer, learning_difficulty_assessor, confidence_scorer],
    tasks=[task1_analyze_requirements, task2_compare_skills, task3_assess_learning_difficulty, task4_calculate_confidence],
    process=Process.sequential,
    verbose=True
)

# --- Run the crew ---
if __name__ == "__main__":
    print("\n########################")
    print("## Starting the Skill Analysis Crew... ##")
    print("########################\n")

    # final_result = crew.kickoff()
    res = llm.invoke("hello")
    print(f"LLM Response: {res}")

    print("\n\n########################")
    print("## Skill Analysis Complete! ##")
    print("########################\n")
    # print(f"Overall Confidence Score: {final_result}%")
