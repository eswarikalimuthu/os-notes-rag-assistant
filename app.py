"""
app.py
------
FastAPI server exposing the RAG pipeline as an HTTP API.

Run:
    uvicorn app:app --reload

Then test with:
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"query": "what is a deadlock", "k": 3}'

Or open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
#import rag          # paid: uses Anthropic Claude API
import rag_free as rag   # free: uses local Ollama instead -- comment the line above
                            # and uncomment this one to run with zero API cost

app = FastAPI(title="OS Notes RAG API", version="1.0")


class AskRequest(BaseModel):
    query: str
    k: int = 3


class Source(BaseModel):
    id: str
    title: str
    distance: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]


@app.get("/")
def health():
    return {"status": "ok", "message": "OS Notes RAG API is running. POST to /ask."}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = rag.ask(req.query, k=req.k)
    return result
