import argparse
import sys
import os
import numpy as np
from core.ollama_client import OllamaClient
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore

class SemanticRetriever:
    def __init__(self, model: str = "nomic-embed-text"):
        self.ollama = OllamaClient(model=model)
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()

    def index_directory(self, path: str):
        """
        Indexes all files in the given directory.
        """
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist.")
            return

        print(f"Indexing directory: {path}...")
        all_chunks = []
        batch_embeddings = []
        
        # Process files and collect chunks
        for item in self.doc_processor.process_directory(path):
            all_chunks.append(item)
            
        if not all_chunks:
            print("No supported files found to index.")
            return

        print(f"Found {len(all_chunks)} chunks. Generating embeddings...")
        
        # Generate embeddings in a loop (Ollama local API is usually sequential)
        # We could optimize this but for local use it's fine.
        embeddings = []
        for i, item in enumerate(all_chunks):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(all_chunks)} chunks embedded...", end='\r')
            
            emb = self.ollama.get_embedding(item["content"])
            embeddings.append(emb)
        
        print(f"\nCompleted embedding generation.")

        # Add to vector store
        self.vector_store.add_documents(np.array(embeddings), all_chunks)
        self.vector_store.save()
        print("Done.")

    def search(self, query: str, top_k: int = 3):
        """
        Performs semantic search for a query.
        """
        if not self.vector_store.load():
            print("No index found. Please run indexing first (--index <path>).")
            return

        print(f"Searching for: \"{query}\"")
        query_emb = self.ollama.get_embedding(query)
        results = self.vector_store.search(query_emb, k=top_k)

        if not results:
            print("No matching results found.")
            return

        print(f"\nTop {len(results)} matches:\n" + "="*50)
        for i, res in enumerate(results, 1):
            score = res["score"]
            content = res["content"].replace('\n', ' ')[:300] + "..."
            path = res["metadata"]["path"]
            
            print(f"{i}. [Score: {score:.4f}] {path}")
            print(f"   Snippet: {content}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Semantic File Retrieval System using local Ollama")
    parser.add_argument("--index", help="Directory path to index")
    parser.add_argument("--query", help="Semantic search query")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama model to use")

    args = parser.parse_args()
    retriever = SemanticRetriever(model=args.model)

    if args.index:
        retriever.index_directory(args.index)
    elif args.query:
        retriever.search(args.query, top_k=args.k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
