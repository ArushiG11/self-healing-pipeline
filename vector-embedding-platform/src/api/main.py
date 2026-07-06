from fastapi import FastAPI
from pydantic import BaseModel
from src.generation.rag import RagPipeline

app = FastAPI(title="Review RAG API")
rag = RagPipeline(table="chunks_sentence")   # the A/B winner — a config choice you can defend with data

class AskRequest(BaseModel):
    question: str
    min_rating: float | None = None

class SourceOut(BaseModel):
    rank: int
    score: float
    rating: float | None
    asin: str | None
    text: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]

@app.get("/health")
def health():
    return {"status": "ok", "chunks": rag.store.count()}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    r = rag.ask(req.question, min_rating=req.min_rating)
    return AskResponse(
        question=r.question,
        answer=r.answer,
        sources=[SourceOut(rank=i+1, score=h.score,
                           rating=h.metadata["rating"], asin=h.metadata["asin"],
                           text=h.document[:200])
                 for i, h in enumerate(r.sources)],
    )