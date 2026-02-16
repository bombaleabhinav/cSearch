import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Adjust imports based on running context (as module vs script)
try:
    from .ingestion import IngestionEngine
    from .indexing import Indexer
    from .search import SearchEngine
    from .embedding import EmbeddingModel
    from .config import INDEX_FILE
except ImportError:
    # Fallback for running directly if needed, though running as module is preferred
    from src.ingestion import IngestionEngine
    from src.indexing import Indexer
    from src.search import SearchEngine
    from src.embedding import EmbeddingModel
    from src.config import INDEX_FILE

app = FastAPI(title="Local Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
# Initialize or load existing index
print("Initializing components...")
indexer = Indexer.load()
if indexer is None:
    print("No existing index found. Creating new indexer.")
    indexer = Indexer()

embedding_model = EmbeddingModel()
search_engine = SearchEngine(indexer)
ingestion_engine = IngestionEngine()

class IngestRequest(BaseModel):
    directory: str
    rebuild: bool = False

class SearchResponse(BaseModel):
    path: str
    score: float
    type: str
    excerpt: str
    semantic_score: float
    lexical_score: float

@app.post("/ingest")
def ingest_files(request: IngestRequest):
    """
    Ingests files from the specified directory.
    If rebuild=True, clears existing index first.
    """
    directory = Path(request.directory)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    if request.rebuild:
        global indexer
        indexer = Indexer()
        # Update search engine reference
        search_engine.indexer = indexer

    files = []
    # Recursively find supported files
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = Path(root) / filename
            if ingestion_engine.is_supported(file_path):
                files.append(file_path)

    if not files:
        return {"message": "No supported files found to ingest.", "count": 0}

    processed_count = 0
    errors = []

    print(f"Found {len(files)} files to process in {directory}...")

    # Process files
    for file_path in files:
        try:
            # Check if file is already indexed (skip if path exists and not rebuild)
            # Basic check: if path in indexer.paths. 
            # Note: paths is a list, O(N) lookup. For 500 files, it's fine.
            if not request.rebuild and str(file_path) in indexer.paths:
                continue

            # Process content
            file_data = ingestion_engine.process_file(file_path)
            if not file_data:
                continue

            # Generate embeddings
            # File level
            file_vec = embedding_model.encode([file_data['representative_text']])[0]
            
            # Chunk level
            chunks = file_data.get('chunks', [])
            chunk_vecs = []
            if chunks:
                chunk_vecs = embedding_model.encode(chunks)
            
            # Add to index
            indexer.add_file(file_data, file_vec, chunk_vecs)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")
            
        except Exception as e:
            msg = f"Error ingesting {file_path}: {str(e)}"
            print(msg)
            errors.append(msg)

    # Build and save index
    if processed_count > 0:
        print("Building index...")
        indexer.build_index()
        print("Saving index...")
        indexer.save()

    return {
        "message": f"Ingestion complete. Processed {processed_count} new files.", 
        "total_files_in_dir": len(files),
        "errors": errors
    }

@app.get("/search", response_model=List[SearchResponse])
def search(q: str, limit: int = 5):
    """
    Semantic search for the query 'q'.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not indexer.file_map:
         raise HTTPException(status_code=503, detail="Index is empty. Please ingest files first.")

    try:
        results = search_engine.search(q, top_k_final=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "index_size": len(indexer.file_map),
        "indexed_paths": len(indexer.paths)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
