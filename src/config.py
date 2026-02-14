import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
INDEX_FILE = BASE_DIR / "index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Heuristics for chunking and representative text
# We use length approximations as tokenization is slow
AVG_CHARS_PER_TOKEN = 4
CHUNKING_MAX_TOKENS = 256
CHUNKING_OVERLAP = 32

REPRESENTATIVE_CONTENT_LIMIT = 2000 # Characters

# Performance Tweaks
BATCH_SIZE = 32

# Scoring Weights (Hybrid)
# score = w1 * semantic + w2 * lexical + w3 * metadata
WEIGHT_SEMANTIC = 0.6
WEIGHT_LEXICAL = 0.3
WEIGHT_METADATA = 0.1

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".html", ".htm", ".json", 
    ".py", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".rs", ".go", ".java"
}
