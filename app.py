from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import ast
import re
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

llm = ChatGroq(
    model_name="groq/llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# === Agents from your flowchart ===

requirement_analyzer = Agent(
    role='Requirement Analyzer 🕵️',
    goal='Analyze non-technical client input and extract the technical skills required for the task',
    backstory="""Specializes in interpreting non-technical project descriptions from clients and identifying the technical tools, libraries, or frameworks needed to implement it.""",
    llm=llm,
    verbose=True
)

skill_comparer = Agent(
    role='Skill Comparer 🤹',
    goal='Compare developer skills with required client skills',
    backstory="""Takes in two skill lists and identifies matches and missing skills.""",
    llm=llm,
    verbose=True
)

relation_mapper = Agent(
    role='Relation Mapper',
    goal='Map missing skills to related or transferable skills and assess difficulty level',
    backstory="""Helps identify if the missing skills can be quickly learned or need external expertise. If the missing and known skills are from similar domains (e.g. Python and Flask), mark them as easy to learn.""",
    llm=llm,
    verbose=True
)

confidence_scorer = Agent(
    role='Confidence Scorer',
    goal='Score the confidence of fulfilling the client requirements',
    backstory="""Uses skill match ratio and mapping to assign a percentage confidence level.""",
    llm=llm,
    verbose=True
)

# === Input data ===

non_technical_description = (
    "I want to build a chatbot that answers customer questions using information from our product manuals. "
    "It should sound smart and respond fast, even when lots of people ask at once."
)

developer_skills = [
    "Python", "React", "AWS", "LangChain", "Docker", "PyTorch", "TensorFlow", "FastAPI", "Flask",
    "PostgreSQL", "MongoDB", "JavaScript", "TypeScript", "Node.js", "Git", "Linux",
    "Kubernetes", "Redis", "GraphQL", "HTML", "CSS"
]

def extract_skills():
    task = Task(
        description=(
            f"""You are an AI assistant that extracts required technical skills, libraries, frameworks, or tools 
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

    crew = Crew(agents=[requirement_analyzer], tasks=[task], verbose=False)
    print("\n🔍 Extracting Required Skills from Client Request...\n")
    result = crew.kickoff()

    raw_output = result.output if hasattr(result, 'output') else str(result)

    # Try to extract a Python list from raw output using regex
    match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
    if match:
        try:
            skills = ast.literal_eval(match.group(0))
        except Exception:
            print("❌ Could not parse the skill list, fallback used.")
            skills = ["LangChain", "Flask", "Kubernetes", "GPT-4"]
    else:
        print("❌ No list found in output, fallback used.")
        skills = ["LangChain", "Flask", "Kubernetes", "GPT-4"]

    print("✅ Extracted Skills:", skills)
    return skills


# === Step 2: Compare developer and client skills ===
def compare_skills(client_skills):
    task = Task(
        description=(
            f"Compare the developer skills: {developer_skills} "
            f"with the client required skills: {client_skills}. "
            f"Identify which skills match and which are missing."
        ),
        expected_output="A JSON or Python dict with keys 'matched_skills' and 'missing_skills' and list values.",
        agent=skill_comparer
    )
    crew = Crew(agents=[skill_comparer], tasks=[task], verbose=False)
    print("\n🤹 Comparing Developer Skills with Client Required Skills...\n")
    result = crew.kickoff()
    raw_output = result.output if hasattr(result, 'output') else str(result)
    
    # Extract matched and missing skills from output (expecting dict or JSON)
    try:
        skills_dict = ast.literal_eval(raw_output)
        matched = skills_dict.get('matched_skills', [])
        missing = skills_dict.get('missing_skills', [])
    except Exception:
        print("❌ Failed to parse comparison output. Using manual set logic.")
        matched = list(set(developer_skills) & set(client_skills))
        missing = list(set(client_skills) - set(developer_skills))

    print("🧠 Matched Skills:", matched)
    print("⚠️ Missing Skills:", missing)
    return matched, missing

# === Step 3: Map relations for missing skills ===
def map_relations(missing_skills):
    task = Task(
        description=(
            f"Given the developer skills: {developer_skills}\n"
            f"And the missing skills: {missing_skills}\n"
            "For each missing skill, explain if it is related to any developer skill and how easy it would be for the developer to learn it. "
            "If a missing skill is a library or framework for a language the developer knows, mark it as 'Easy to Learn'. "
            "If there is overlap, explain the relationship. Otherwise, say 'Assess manually'."
        ),
        expected_output="Mapping of missing skills to related skills and difficulty.",
        agent=relation_mapper
    )
    crew = Crew(agents=[relation_mapper], tasks=[task], verbose=False)
    print("\n🔗 Mapping Relations for Missing Skills...\n")
    result = crew.kickoff()
    mapping_text = result.output if hasattr(result, 'output') else str(result)
    print(mapping_text)
    return mapping_text

# === Step 4: Confidence score based on mapping ===
def score_confidence(matched_skills, total_skills, mapping_text):
    task = Task(
        description=(
            f"Using matched skills count {len(matched_skills)} and total required skills {total_skills}, "
            f"and the following mapping:\n{mapping_text}\n"
            "Provide a confidence score (0-100%) on the ability to fulfill client requirements."
        ),
        expected_output="Confidence score as a percentage.",
        agent=confidence_scorer
    )
    crew = Crew(agents=[confidence_scorer], tasks=[task], verbose=False)
    print("\n⭐ Calculating Confidence Score...\n")
    result = crew.kickoff()
    confidence_str = result.output if hasattr(result, 'output') else str(result)
    print(confidence_str)
    try:
        # Extract integer from response
        score = int(re.search(r'\d+', confidence_str).group())
    except Exception:
        score = int((len(matched_skills) / total_skills) * 100)
    print(f"📊 Confidence Score: {score}%")
    return score


if __name__ == "__main__":
    # Step 1
    client_skills = extract_skills()
    
    # Step 2
    matched_skills, missing_skills = compare_skills(client_skills)
    
    # Steps 3 & 4 run in parallel
    with ThreadPoolExecutor() as executor:
        future_mapping = executor.submit(map_relations, missing_skills)
        # Wait for mapping result to pass to confidence scorer
        mapping_text = future_mapping.result()
        future_confidence = executor.submit(score_confidence, matched_skills, len(client_skills), mapping_text)
        confidence_score = future_confidence.result()
