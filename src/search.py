import numpy as np
from typing import List, Dict
from .indexing import Indexer
from .embedding import EmbeddingModel
from .config import WEIGHT_SEMANTIC, WEIGHT_LEXICAL, WEIGHT_METADATA

class SearchEngine:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.model = EmbeddingModel()

    def search(self, query: str, top_k_initial: int = 10, top_k_final: int = 3) -> List[Dict]:
        """
        Performs two-stage retrieval:
        1. File-level semantic filtering (Global + BM25 optional)
        2. Chunk-level verification on top K
        3. Hybrid scoring
        """
        # 1. Embed Query
        query_vec = self.model.encode([query])[0]
        
        # 2. Global Semantic Search
        # (N,) array
        global_scores = self.indexer.matrix @ query_vec 
        
        # 3. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.indexer.bm25.get_scores(tokenized_query)
        # Normalize BM25 (0-1)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
            
        # 4. Candidate Selection
        # We short-list based on Global Semantic Score primary, maybe + BM25?
        # User said: "File-level embeddings... PRIMARY retrieval step"
        # Let's use a mix for filtering to be safe, or just global.
        # User: "Perform cosine similarity search on file embeddings. Select top K... candidate files"
        # Implies just semantic for filtering.
        initial_indices = np.argsort(global_scores)[-top_k_initial:][::-1]
        
        results = []
        for idx in initial_indices:
            file_data = self.indexer.file_map[idx]
            global_sem_score = float(global_scores[idx])
            
            # Chunk Search
            chunks = file_data.get('chunks', [])
            chunk_vecs = np.array([c['vector'] for c in chunks]) if chunks else np.array([])
            
            best_chunk_idx = -1
            best_chunk_score = 0.0
            
            if len(chunk_vecs) > 0:
                chunk_scores = chunk_vecs @ query_vec
                best_chunk_idx = int(np.argmax(chunk_scores))
                best_chunk_score = float(chunk_scores[best_chunk_idx])
            else:
                best_chunk_score = global_sem_score # Fallback
                
            # Combined Semantic Score (Global + Local)
            # Giving weight to both ensures we match file intent AND specific content
            semantic_score = 0.4 * global_sem_score + 0.6 * best_chunk_score
            
            # Lexical Score
            lexical_score = float(bm25_scores[idx])
            
            # Metadata Score (e.g. filename match)
            meta_score = 0.0
            if query.lower() in file_data['path'].lower(): # Simple substring match
                meta_score = 1.0
            
            # Final Hybrid Score
            final_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_LEXICAL * lexical_score +
                WEIGHT_METADATA * meta_score
            )
            
            results.append({
                'path': file_data['path'],
                'score': final_score,
                'type': file_data['type'],
                'excerpt': chunks[best_chunk_idx]['text'] if best_chunk_idx != -1 else file_data['representative_text'][:200],
                'semantic_score': semantic_score,
                'lexical_score': lexical_score
            })
            
        # Sort and return top K
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k_final]
