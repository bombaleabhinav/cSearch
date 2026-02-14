import pickle
import numpy as np
import os
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from .config import INDEX_FILE
from .utils import normalize_vector

class Indexer:
    def __init__(self):
        self.file_map = {}  # int_id -> file_data
        self.file_vectors = [] # List of numpy arrays
        self.bm25 = None
        self.corpus_tokens = []
        self.paths = [] # To map index back to file path/id
        self.matrix = None

    def add_file(self, file_data: Dict, file_embedding: np.ndarray, chunk_embeddings: np.ndarray):
        """
        Adds a file to the index.
        """
        fid = len(self.file_map)
        
        # Store chunks with their embeddings
        chunks_with_vecs = []
        # Ensure chunk embeddings list matches chunks list length
        if len(file_data.get('chunks', [])) != len(chunk_embeddings):
            # Fallback or truncate?
            print(f"Warning: Chunk mismatch for {file_data.get('path')}. Text: {len(file_data['chunks'])}, Embeds: {len(chunk_embeddings)}")
            # If so, zip safest length
            limit = min(len(file_data['chunks']), len(chunk_embeddings))
            for i in range(limit):
                chunks_with_vecs.append({
                    'text': file_data['chunks'][i],
                    'vector': chunk_embeddings[i]
                })
        else:
            for i, text in enumerate(file_data['chunks']):
                chunks_with_vecs.append({
                    'text': text,
                    'vector': chunk_embeddings[i]
                })
            
        self.file_map[fid] = {
            'path': file_data['path'],
            'type': file_data['type'],
            'metadata': file_data['metadata'],
            'representative_text': file_data['representative_text'],
            'chunks': chunks_with_vecs
        }
        
        # Normalize and store file vector
        vec = normalize_vector(file_embedding)
        self.file_vectors.append(vec)
        
        # Prepare for BM25 (using representative text + some content)
        # Using representative text for BM25 might be better than full text for speed/noise.
        # But for Lexical search, maybe full text is better?
        # User constraint: Accuracy > Recall.
        # Let's use representative text + first 5 chunks as proxy for "content"
        # If chunks are small?
        text_for_bm25 = file_data['representative_text'] + " " + " ".join(file_data['chunks'][:5])
        tokens = text_for_bm25.lower().split()
        self.corpus_tokens.append(tokens)
        self.paths.append(file_data['path'])

    def build_index(self):
        """Finalizes the index structures."""
        if not self.file_vectors:
            print("No files to index.")
            return

        self.matrix = np.vstack(self.file_vectors)
        print(f"Building BM25 index for {len(self.corpus_tokens)} documents...")
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Clear corpus tokens to save memory
        self.corpus_tokens = [] 

    def save(self):
        print(f"Saving index to {INDEX_FILE}...")
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump({
                #'file_map': self.file_map, # Wait, file_map can be huge if chunks are huge?
                # The user wants "Cache all embeddings permanently".
                # pickle is easy.
                'file_map': self.file_map,
                'matrix': self.matrix,
                'bm25': self.bm25,
                'paths': self.paths
            }, f)
        print("Index saved.")

    @classmethod
    def load(cls):
        if not os.path.exists(INDEX_FILE):
            print("No index found.")
            return None
            
        print(f"Loading index from {INDEX_FILE}...")
        try:
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
            
            indexer = cls()
            indexer.file_map = data['file_map']
            indexer.matrix = data['matrix']
            indexer.bm25 = data['bm25']
            indexer.paths = data['paths']
            print(f"Index loaded with {len(indexer.file_map)} files.")
            return indexer
        except Exception as e:
            print(f"Failed to load index: {e}")
            return None
