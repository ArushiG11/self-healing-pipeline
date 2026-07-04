from abc import ABC, abstractmethod

class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def dim(self) -> int: ...