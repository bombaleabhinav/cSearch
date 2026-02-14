from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch
from .config import MODEL_NAME, BATCH_SIZE

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading embedding model {MODEL_NAME} on {device}...")
            cls._instance.model = SentenceTransformer(MODEL_NAME, device=device)
        return cls._instance

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Generates normalized embeddings for a list of texts.
        """
        if not texts:
            return np.array([])
            
        embeddings = self.model.encode(
            texts, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=show_progress, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        return embeddings
