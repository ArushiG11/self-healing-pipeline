from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Hit:
    id: str
    score: float
    document: str
    metadata: dict

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids, vectors, documents, metadatas) -> None: ...
    @abstractmethod
    def query(self, vector, k: int = 5, filter: dict | None = None) -> list[Hit]: ...
    @abstractmethod
    def count(self) -> int: ...