import requests
import numpy as np
from typing import List

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model = model
        self.endpoint = f"{base_url}/api/embeddings"

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Fetches embedding for a single text string from Ollama.
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding, dtype="float32")
        except Exception as e:
            print(f"Error fetching embedding from Ollama: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Note: Ollama API typically processes one at a time via the standard endpoint,
        but we wrap it for potential future batch optimization or manual loop.
        """
        return [self.get_embedding(text) for text in texts]
