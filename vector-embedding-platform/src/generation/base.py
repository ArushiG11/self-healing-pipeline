from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...