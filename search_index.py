import requests
import numpy as np
import faiss
import pickle

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"


def get_embedding(text: str) -> np.ndarray:
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype="float32")


def search(query: str, k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    qvec = get_embedding(query).reshape(1, -1)
    D, I = index.search(qvec, k)

    print("\nResults:")
    shown = set()
    for idx in I[0]:
        path = meta[idx]
        if path not in shown:
            print(path)
            shown.add(path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python search_index.py \"your search query\"")
        exit(1)

    search(sys.argv[1])