from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from canvas_api import search_canvas
from summarizer import summarize_text, extract_em_text

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchRequest(BaseModel):
    keyword: str


@app.post("/search")
async def search(req: SearchRequest):
    session = search_canvas(req.keyword)
    if "content" not in session:
        raise HTTPException(status_code=500, detail="Canvas search failed")

    raw_content = session.get("content", "")

    # ✅ Extract <em> if available, else fallback to plain text
    cleaned_text = extract_em_text(raw_content)

    summary = summarize_text(cleaned_text)

    return {
        "session_title": session.get("session") or session.get("title"),
        "resource_title": session.get("title"),
        "summary": summary
    }