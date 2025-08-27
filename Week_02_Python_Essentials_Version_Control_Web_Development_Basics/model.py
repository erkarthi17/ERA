import re
import openai
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file if present
load_dotenv()

# Fetch the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------
# Internal Analysis Functions
# ----------------------------
def analyze_emotion_internal(text: str) -> str:
    text_lower = text.lower()
    critical_keywords = ["crash", "failure", "immediately", "urgent", "critical", "asap", "emergency", "not working"]
    high_keywords = ["slow", "delay", "issue", "problem", "error", "bug", "important"]
    low_keywords = ["suggestion", "feature request", "enhancement", "nice to have", "minor"]

    if any(word in text_lower for word in critical_keywords):
        return "critical"
    elif "urgent" in text_lower:
        return "urgent"
    elif any(word in text_lower for word in high_keywords):
        return "high"
    elif any(word in text_lower for word in low_keywords):
        return "low"
    else:
        return "neutral"

def extract_theme_internal(text: str) -> str:
    if re.search(r"\b(login|authentication|password)\b", text, re.I):
        return "Authentication Issue"
    elif re.search(r"\b(crash|error|failure|bug)\b", text, re.I):
        return "System Crash"
    elif re.search(r"\b(network|connect|internet|server)\b", text, re.I):
        return "Connectivity Issue"
    elif re.search(r"\b(ui|display|screen|layout)\b", text, re.I):
        return "UI/UX Problem"
    else:
        return "General Issue"

# ----------------------------
# OpenAI Analysis Function
# ----------------------------
def analyze_with_openai(problem_text: str) -> dict:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    Analyze the following user-submitted problem and return:
    - Emotion (one of: critical, urgent, high, neutral, low)
    - Theme (e.g., Authentication Issue, System Crash, Connectivity Issue, UI/UX Problem, General Issue)

    Problem: "{problem_text}"

    Respond ONLY in JSON format with keys: emotion, theme.
    """

    logging.info(f"Sending prompt to OpenAI:\n{prompt}")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # use GPT-3.5 by default for safety
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Extract content
        content = response.choices[0].message["content"].strip()
        # Remove markdown code blocks if present
        clean_content = re.sub(r"^```json|```$", "", content, flags=re.MULTILINE).strip()

        logging.info(f"Received response from OpenAI:\n{clean_content}")
        return json.loads(clean_content)

    except Exception as e:
        logging.error(f"OpenAI analysis failed: {e}. Falling back to internal analysis.")
        return {
            "emotion": analyze_emotion_internal(problem_text),
            "theme": extract_theme_internal(problem_text)
        }

# ----------------------------
# Bulk Processing Function
# ----------------------------
def analyze_bulk_problems(problems: list) -> list:
    results = []
    for problem in problems:
        logging.info(f"Analyzing problem: {problem}")
        result = analyze_with_openai(problem)
        results.append({
            "problem": problem,
            "emotion": result.get("emotion"),
            "theme": result.get("theme")
        })
    return results

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    sample_problems = [
        "All the systems are not working at the office, need a solution ASAP",
        "Login page keeps failing when I enter my password",
        "The new UI layout is confusing and buttons overlap on small screens",
        "It would be nice to have a dark mode option",
        "Internet connectivity is unstable and disconnects frequently"
    ]

    analysis_results = analyze_bulk_problems(sample_problems)
    print(json.dumps(analysis_results, indent=4))
