import os
import requests
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from PyPDF2 import PdfReader
from docx import Document

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"
CHUNK_SIZE = 800


def get_embedding(text: str) -> np.ndarray:
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype="float32")


def extract_text(path: str) -> str:
    try:
        if path.endswith(".txt"):
            return open(path, encoding="utf-8", errors="ignore").read()
        if path.endswith(".pdf"):
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        if path.endswith(".docx"):
            return "\n".join(p.text for p in Document(path).paragraphs)
    except Exception:
        return ""
    return ""


def chunk_text(text: str):
    for i in range(0, len(text), CHUNK_SIZE):
        yield text[i:i + CHUNK_SIZE]


def build_index(folder: str):
    vectors = []
    meta = []

    print(f"Indexing files in: {folder}\n")

    for root, _, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith((".txt", ".pdf", ".docx")):
                continue

            path = os.path.join(root, file)
            text = extract_text(path)
            if not text.strip():
                continue

            for chunk in chunk_text(text):
                vec = get_embedding(chunk)
                vectors.append(vec)
                meta.append(path)

    if not vectors:
        print("No valid files found.")
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vectors))

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n Indexed {len(vectors)} chunks.")
    print("Saved:", INDEX_FILE, META_FILE)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python build_index.py <folder>")
        exit(1)

    build_index(sys.argv[1])
