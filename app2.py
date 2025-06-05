from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import ast
import re

load_dotenv()

llm = ChatGroq(
    model_name="groq/llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

non_technical_description = (
    "I want to build a chatbot that answers customer questions using information from our product manuals. "
    "It should sound smart and respond fast, even when lots of people ask at once."
)

developer_skills = [
    "Python", "React", "AWS", "LangChain", "Docker", "PyTorch", "TensorFlow", "FastAPI", "Flask",
    "PostgreSQL", "MongoDB", "JavaScript", "TypeScript", "Node.js", "Git", "Linux",
    "Kubernetes", "Redis", "GraphQL", "HTML", "CSS", "machine learning"
]

requirement_analyzer = Agent(
    role='Requirement Analyzer 🕵️',
    goal='Analyze non-technical client input and extract the technical skills required for the task',
    backstory="""Specializes in interpreting non-technical project descriptions from clients and identifying the technical tools, libraries, or frameworks needed to implement it.
    Also identify the skills from the result that same skills are same or not in the clients request list.""",
    llm=llm,
    verbose=True
)

skill_comparer = Agent(
    role='Skill Comparer 🤹',
    goal=(
        "Compare developer skills with required client skills. "
        "Categorize the skills into: exact matches, similar skills, and missing skills. "
        "For example, if the developer knows Python, it should be considered a similar skill to Machine Learning."
    ),
    backstory=(
        "Takes two lists: required client skills and developer skills. "
        "Identifies direct matches, then uses common industry knowledge to detect closely related or complementary skills. "
        "Provides a breakdown into three categories:\n"
        "- matched_skills (exact matches)\n"
        "- similar_skills (e.g. Python ≈ Machine Learning)\n"
        "- missing_skills (no match or related skill found)"
    ),
    llm=llm,
    verbose=True
)

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

if __name__ == "__main__":
    task1 = Task(
        description=(
            f"""extracts  technical skills, libraries, frameworks, or tools 
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
    crew1 = Crew(agents=[requirement_analyzer], tasks=[task1], verbose=False)
    print("\n🔍 Extracting Required Skills from Client Request...\n")
    result1 = crew1.kickoff()
    raw_output1 = result1.output if hasattr(result1, 'output') else str(result1)
    match = re.search(r'\[.*?\]', raw_output1, re.DOTALL)
    if match:
        try:
            client_skills = ast.literal_eval(match.group(0))
            print("✅ Extracted Skills:", client_skills)
        except Exception:
            print("❌ Could not parse the skill list. Not found skills.")
    else:
        print("❌ No list found in output. Not found skills.")
        
    #------------TASK2-------------
    if isinstance(client_skills, list):
        task2 = Task(
            description=(
                f"""You are given the developer's skills:\n{developer_skills}\n
                And the required client skills:\n{client_skills}\n
                
                Perform a deep comparison using technical reasoning. Go beyond exact matches.
                
                - Consider related or prerequisite skills as "similar skills". For example:
                    • 'Python' ≈ 'LangChain', 'Machine Learning'
                    • 'Machine Learning' ≈ 'NLP', 'PyTorch'
                    • 'Docker' ≈ 'Kubernetes'
                
                - Output a dictionary with the following keys:
                    'matched_skills': skills that exactly match
                    'similar_skills': developer skills that are closely related or commonly used with client skills
                    'missing_skills': client skills that are neither matched nor reasonably inferred

                Return only a Python dict like:
                {{
                    'matched_skills': [...],
                    'similar_skills': [...],
                    'missing_skills': [...]
                }}
                
                Be concise. Do not explain. Just return the Python dict."""
            ),
            expected_output="A Python dict with keys 'matched_skills', 'similar_skills', and 'missing_skills'.",
            agent=s kill_comparer
        )

        crew2 = Crew(agents=[skill_comparer], tasks=[task2], verbose=False)
        print("\n⚁ Comparing Developer Skills with Client Required Skills (Smart Matching)...\n")
        result2 = crew2.kickoff()
        raw_output2 = result2.output if hasattr(result2, 'output') else str(result2)

        try:
            skills_dict = ast.literal_eval(raw_output2)
            matched_skills = skills_dict.get('matched_skills', [])
            similar_skills = skills_dict.get('similar_skills', [])
            missing_skills = skills_dict.get('missing_skills', [])
            # Combine matched and similar skills
            all_matched_skills = list(set(matched_skills + similar_skills))
        except Exception:
            print("❌ Failed to parse smart comparison. Reverting to basic matching.")
            all_matched_skills = list(set(developer_skills) & set(client_skills))
            missing_skills = list(set(client_skills) - set(developer_skills))

        print("✅ Matched Skills (including similar):", all_matched_skills)
        print("⚠️ Missing Skills:", missing_skills)

        # ------------TASK3-------------
        task3 = Task(
            description=(
                f"""Given the developer's skills:\n{developer_skills}\n
                And the missing skills:\n{missing_skills}\n

                For each missing skill, analyze how easy or difficult it would be for the developer to learn it,
                based on their existing skills. Consider overlap, prerequisites, and domain similarity.

                Return a Python dict mapping each missing skill to one of: 'Easy', 'Moderate', or 'Difficult'.

                Example:
                {{
                    'Kubernetes': 'Easy',
                    'GPT-4': 'Moderate'
                    'Flutter': 'Difficult'
                }}

                Do not explain. Just return the Python dict."""
            ),
            expected_output="A Python dict mapping each missing skill to 'Easy', 'Moderate', or 'Difficult'.",
            agent=relation_mapper
        )

        crew3 = Crew(agents=[relation_mapper], tasks=[task3], verbose=False)
        print("\n🧑‍🏫 Assessing Learning Difficulty for Missing Skills...\n")
        result3 = crew3.kickoff()
        raw_output3 = result3.output if hasattr(result3, 'output') else str(result3)
        try:
            difficulty_dict = ast.literal_eval(raw_output3)
        except Exception:
            print("❌ Failed to parse learning difficulty output.")
            difficulty_dict = {}
        print("📚 Learning Difficulty Assessment:", difficulty_dict)

        # ------------TASK4-------------
        task4 = Task(
            description=(
                f"""You are given:
Matched skills (including similar): {all_matched_skills}
Learning difficulty assessment: {difficulty_dict}

Instructions:
1. Count the total number of required skills (matched + missing).
2. Calculate the percentage of matched skills: (number of matched skills / total required skills) * 100.
3. Calculate the percentage of missing skills that are 'Easy' to learn: (number of 'Easy' / total required skills) * 100. Add this to the score.
4. Calculate the percentage of missing skills that are 'Difficult' to learn: (number of 'Difficult' / total required skills) * 100. Subtract this from the score.
5. The final confidence score is: matched percentage + easy percentage - difficult percentage.
6. Return ONLY the final integer percentage. No explanation, no thoughts, no text, just the number.
"""
            ),
            expected_output="An integer percentage representing confidence.",
            agent=confidence_scorer
        )

        crew4 = Crew(agents=[confidence_scorer], tasks=[task4], verbose=False)
        print("\n🏆 Calculating Confidence Score...\n")
        result4 = crew4.kickoff()
        confidence_str = result4.output if hasattr(result4, 'output') else str(result4)
        # Extract the first integer from the output
        match = re.search(r'\d+', confidence_str)
        if match:
            confidence_score = int(match.group())
            print(f"🔢 Confidence Score: {confidence_score}%")
        else:
            print(f"❌ Could not extract a confidence score. Raw output: {confidence_str}")
        
        # ------------CONFIDENCE SCORE (Python logic, not LLM)-------------
        print("\n🏆 Calculating Confidence Score (Python logic)...\n")
        total_required = len(all_matched_skills) + len(missing_skills)
        if total_required == 0:
            confidence_score = 0
        else:
            matched_percentage = (len(all_matched_skills) / total_required) * 100
            easy_count = sum(1 for v in difficulty_dict.values() if v == 'Easy')
            difficult_count = sum(1 for v in difficulty_dict.values() if v == 'Difficult')
            easy_percentage = (easy_count / total_required) * 100
            difficult_percentage = (difficult_count / total_required) * 100
            confidence_score = int(matched_percentage + easy_percentage - difficult_percentage)
            confidence_score = max(0, min(confidence_score, 100))  # Clamp between 0 and 100

        print(f"🔢 Confidence Score: {confidence_score}%")