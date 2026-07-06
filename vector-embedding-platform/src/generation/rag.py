import logging
import numpy as np
from dataclasses import dataclass
from src.embedding.bge import BgeEmbeddingClient
from src.vectorstore.pg import PgVectorStore
from src.generation.groq_client import GroqClient

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag")

SYSTEM = (
    "You answer shopper questions using ONLY the review excerpts provided. "
    "Rules: (1) If the excerpts do not contain enough information to answer, "
    "say 'The reviews I found don't answer this' - never guess. "
    "(2) Cite excerpts by their number like [1] or [2] after each claim. "
    "(3) If reviews disagree, say so and present both sides. "
    "(4) Be concise: 2-5 sentences."
)

@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list  # the Hit objects used

class RagPipeline:
    def __init__(self, table: str = "chunks", k: int = 5):
        self.embedder = BgeEmbeddingClient()
        self.store = PgVectorStore(table=table)
        self.llm = GroqClient()
        self.k = k

    # embed the question (with the bge query prefix, via embed_query),
    # retrieve top-k,
    # format chunks as a numbered context block,
    # prompt the LLM with question + context, 
    # return answer plus the raw sources so callers (your API, your UI) can render citations.

    def ask(self, question: str, min_rating: float | None = None) -> RagAnswer:
        vec = np.array(self.embedder.embed(question))
        f = {"min_rating": min_rating} if min_rating else None
        hits = self.store.query(vec, k=self.k, filter=f)

        context = "\n\n".join(
            f"[{i+1}] (rating {h.metadata['rating']}) {h.document}"
            for i, h in enumerate(hits)
        )
        user = f"Review excerpts:\n{context}\n\nQuestion: {question}"
        answer = self.llm.complete(SYSTEM, user)

        log.info("ask | q=%r hits=%d top_score=%.3f",
                 question[:60], len(hits), hits[0].score if hits else 0.0)
        return RagAnswer(question=question, answer=answer, sources=hits)

if __name__ == "__main__":
    rag = RagPipeline()
    for q in ["What's the best air fryer for a family of six?"
            #   "Is there a good camera bag that protects gear?"
              ]:
        r = rag.ask(q)
        print(f"\nQ: {r.question}\nA: {r.answer}\n")
        for i, h in enumerate(r.sources, 1):
            print(f"  [{i}] rating={h.metadata['rating']} asin={h.metadata['asin']} | {h.document[:80]}")