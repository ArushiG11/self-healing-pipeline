import os
from groq import Groq
from dotenv import load_dotenv
from src.generation.base import LLMClient

load_dotenv()

class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            # for grounded Q&A you want the model boringly faithful to context. We'll drop temperature from 0.7 to 0.2.
        )
        return resp.choices[0].message.content