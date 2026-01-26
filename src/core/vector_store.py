import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, index_path: str = "storage/index.faiss", meta_path: str = "storage/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata = []
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalizes vectors for cosine similarity using Inner Product index.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Adds embeddings and their metadata to the store.
        """
        if len(embeddings) == 0:
            return

        dim = embeddings.shape[1]
        normalized_embeddings = self._normalize(embeddings)

        if self.index is None:
            # We use IndexFlatIP for Inner Product (cosine similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(normalized_embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Searches for the top k most similar documents.
        Returns a list of results with scores and metadata.
        """
        if self.index is None:
            return []

        # Normalize query embedding
        query_vec = query_embedding.reshape(1, -1)
        query_vec = self._normalize(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            result = {
                "score": float(score),
                "content": self.metadata[idx]["content"],
                "metadata": self.metadata[idx]["metadata"]
            }
            results.append(result)
        
        return results

    def save(self):
        """
        Persists the index and metadata to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, 'wb') as f:
                pickle.dump({"metadata": self.metadata}, f)
            print(f"Saved index and metadata to {os.path.dirname(self.index_path)}")

    def load(self):
        """
        Loads the index and metadata from disk.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
            return True
        return False
