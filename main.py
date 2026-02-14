import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

# Increase recursion depth just in case
sys.setrecursionlimit(10000)

# Add src to path if running from root
sys.path.append(str(Path(__file__).parent))

from src.ingestion import IngestionEngine
from src.embedding import EmbeddingModel
from src.indexing import Indexer
from src.search import SearchEngine

def ingest(directory: str):
    print(f"Starting ingestion for: {directory}")
    start_time = time.time()
    
    ingestion_engine = IngestionEngine()
    indexer = Indexer()
    embedding_model = EmbeddingModel()
    
    # 1. Walk directory
    files_to_process = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = Path(root) / file
            if ingestion_engine.is_supported(path):
                files_to_process.append(path)
                
    print(f"Found {len(files_to_process)} supported files.")
    if not files_to_process:
        return

    # 2. Process and Embed
    # We batch process embeddings for efficiency if possible, 
    # but our architecture processes file-by-file for simplicity first.
    # To optimize: collect text chunks and embed in batches. 
    # But for "Heterogeneous" files, each file generates multiple chunks.
    
    # Let's process file -> (text, chunks) -> embed -> index
    
    # Pre-loading model outside loop
    _ = embedding_model
    
    fail_count = 0
    success_count = 0
    
    with tqdm(total=len(files_to_process), desc="Ingesting") as pbar:
        for file_path in files_to_process:
            try:
                # A. Parse & Chunk
                file_data = ingestion_engine.process_file(file_path)
                if not file_data:
                    fail_count += 1
                    pbar.update(1)
                    continue
                
                # B. Embed File Representative Text
                # Just 1 text
                file_vec = embedding_model.encode([file_data['representative_text']])[0]
                
                # C. Embed Chunks
                # If chunks exist
                chunks = file_data.get('chunks', [])
                chunk_vecs = []
                if chunks:
                    chunk_vecs = embedding_model.encode(chunks)
                
                # D. Add to Index
                indexer.add_file(file_data, file_vec, chunk_vecs)
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                fail_count += 1
            
            pbar.update(1)

    # 3. Finalize & Save
    indexer.build_index()
    indexer.save()
    
    elapsed = time.time() - start_time
    print(f"Ingestion complete in {elapsed:.2f}s.")
    print(f"Indexed: {success_count}, Failed: {fail_count}")

def search(query: str):
    # Load Index
    indexer = Indexer.load()
    if not indexer:
        print("Please run ingestion first.")
        return

    search_engine = SearchEngine(indexer)
    start_time = time.time()
    
    results = search_engine.search(query, top_k_initial=10, top_k_final=5)
    
    elapsed = time.time() - start_time
    print(f"\nSearch results for: '{query}' ({elapsed:.4f}s)")
    print("-" * 60)
    
    if not results:
        print("No matches found.")
        return

    # Print top result detailed, others summary
    # User requested: "Return the BEST matching file (not a chunk)"
    # But also "Output requirements: Return file path, file type, and relevance score"
    
    top = results[0]
    print(f"TOP MATCH: {top['path']}")
    print(f"Type: {top['type'].upper()} | Score: {top['score']:.4f}")
    print(f"Excerpt: \"{top['excerpt'].strip()}\"\n")
    
    if len(results) > 1:
        print("Other Candidates:")
        for res in results[1:]:
            print(f"  [{res['score']:.4f}] {Path(res['path']).name} ({res['type']})")

def main():
    parser = argparse.ArgumentParser(description="Local Semantic File Search")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Ingest Command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest files from a directory')
    ingest_parser.add_argument('directory', type=str, help='Path to directory')
    
    # Search Command
    search_parser = subparsers.add_parser('search', help='Search for files')
    search_parser.add_argument('query', type=str, help='Search query')
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        ingest(args.directory)
    elif args.command == 'search':
        search(args.query)

if __name__ == "__main__":
    main()
