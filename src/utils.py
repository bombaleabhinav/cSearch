import re
import numpy as np

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 norm)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Assuming input vectors might be batched or single
    # Dot product because vectors are normalized
    return np.dot(vec1, vec2)
