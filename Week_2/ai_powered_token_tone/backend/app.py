from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from model import analyze_emotion_internal, extract_theme_internal, analyze_with_openai

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI is live on EC2!"}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ProblemRequest(BaseModel):
    problem: str

@app.post("/analyze")
async def analyze_problem(request: ProblemRequest):
    problem = request.problem
    emotion = None
    theme = None
    source_id = None

    try:
        logging.info("Calling OpenAI for problem analysis...")
        llm_result = analyze_with_openai(problem)
        emotion = llm_result.get("emotion")
        theme = llm_result.get("theme")

        if emotion and theme:
            source_id = "OPENAI"
            logging.info(f"OpenAI returned emotion: {emotion}, theme: {theme}")
        else:
            logging.warning("OpenAI returned incomplete data, using internal analysis")
            emotion = analyze_emotion_internal(problem)
            theme = extract_theme_internal(problem)
            source_id = "INTERNAL FALLBACK"

    except Exception as e:
        logging.error(f"OpenAI call failed: {e}, using internal analysis")
        emotion = analyze_emotion_internal(problem)
        theme = extract_theme_internal(problem)
        source_id = "INTERNAL FALLBACK"

    # Map emotion to priority
    priority_map = {
        "critical": "P1",
        "urgent": "P1",
        "high": "P2",
        "neutral": "P3",
        "low": "P4"
    }
    priority = priority_map.get(emotion.lower(), "P3")
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    response = {
        "ticket_id": ticket_id,
        "theme": theme,
        "emotion": emotion.capitalize(),
        "priority": priority,
        "source": source_id,
        "message": f"We understand your problem. Considering the statement given, we have raised a {priority} ticket."
    }

    return response
