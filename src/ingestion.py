import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm
from .config import REPRESENTATIVE_CONTENT_LIMIT, SUPPORTED_EXTENSIONS, CHUNKING_MAX_TOKENS, AVG_CHARS_PER_TOKEN, CHUNKING_OVERLAP
from .utils import clean_text

class IngestionEngine:
    """Handles file reading, representative text extraction, and semantic chunking."""

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def process_file(self, file_path: Path) -> Dict:
        """
        Parses a file and returns:
          - full_text: Raw or slightly cleaned text
          - representative_text: Text used for file-level embedding
          - chunks: List of chemically meaningful text blocks
          - metadata: Minimal metadata dict
        """
        ext = file_path.suffix.lower()
        try:
            if ext == '.pdf':
                return self._process_pdf(file_path)
            elif ext in ['.html', '.htm']:
                return self._process_html(file_path)
            elif ext in ['.py', '.js', '.ts', '.c', '.cpp', '.h', '.cs', '.go', '.rs', '.java']:
                return self._process_code(file_path)
            else:
                return self._process_text(file_path)
        except Exception as e:
            # print(f"Error processing {file_path}: {e}") # Silent fail or log?
            return None

    def _process_pdf(self, path: Path) -> Dict:
        reader = PdfReader(str(path))
        full_text_list = []
        representative_parts = []
        
        # Extract metadata if available
        meta = reader.metadata
        if meta:
            if meta.title: representative_parts.append(str(meta.title))
            if meta.subject: representative_parts.append(str(meta.subject))

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_str = str(text)
                full_text_list.append(text_str) 
                # Heuristic: First few lines are often headers
                lines = text_str.split('\n')
                representative_parts.append(str(lines[0]) if lines else "") # Safely add first line 

        raw_full_text = "\n\n".join(full_text_list) # Use double newline to separate pages for chunking
        
        representative_text = " ".join(representative_parts[:20]) 
        representative_text = clean_text(representative_text)[:REPRESENTATIVE_CONTENT_LIMIT]

        chunks = self._chunk_text(raw_full_text)
        
        return {
            "path": str(path),
            "type": "pdf",
            "representative_text": representative_text if representative_text else clean_text(raw_full_text)[:REPRESENTATIVE_CONTENT_LIMIT],
            "chunks": chunks,
            "metadata": {"title": str(meta.title) if meta and meta.title else ""}
        }

    def _process_html(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()

        if soup.title and soup.title.string: 
            title_str = str(soup.title.string)
            rep_parts.append(title_str)
        
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"): 
            rep_parts.append(str(meta_desc.get("content")))
        
        for h in soup.find_all(['h1', 'h2', 'h3']):
            rep_parts.append(h.get_text())

        # For chunking, get text with separators to preserve structure?
        # get_text(separator='\n') helps chunking
        full_text = soup.get_text(separator='\n\n')
        
        representative_text = " ".join(rep_parts)
        representative_text = clean_text(representative_text)[:REPRESENTATIVE_CONTENT_LIMIT]

        chunks = self._chunk_text(full_text)
        
        title_meta = ""
        if soup.title and soup.title.string:
            title_meta = str(soup.title.string)

        return {
            "path": str(path),
            "type": "html",
            "representative_text": representative_text if representative_text else clean_text(full_text)[:REPRESENTATIVE_CONTENT_LIMIT],
            "chunks": chunks,
            "metadata": {"title": title_meta}
        }

    def _process_code(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Representative: Filename, Docstrings/Comments, Function/Class definitions
        rep_parts = [path.name]
        
        defs = re.findall(r'(?:class|def|function|void|int|str|public|private)\s+([a-zA-Z_]\w*)', content)
        rep_parts.extend(defs[:20])

        comments = re.findall(r'(?://|#)\s*(.*)', content)
        rep_parts.extend(comments[:10])

        representative_text = " ".join(rep_parts)
        representative_text = clean_text(representative_text)[:REPRESENTATIVE_CONTENT_LIMIT]
        
        return {
            "path": str(path),
            "type": "code",
            "representative_text": representative_text,
            "chunks": self._chunk_text(content, is_code=True),
            "metadata": {}
        }

    def _process_text(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        representative_text = clean_text(content)[:REPRESENTATIVE_CONTENT_LIMIT]

        return {
            "path": str(path),
            "type": "text",
            "representative_text": representative_text,
            "chunks": self._chunk_text(content),
            "metadata": {}
        }

    def _chunk_text(self, text: str, is_code: bool = False) -> List[str]:
        """
        Semantic chunking:
        1. Split by double newlines (paragraphs).
        2. If chunks are too large, split by sentences/newlines.
        3. Merge small chunks to target size.
        """
        # Target chars per chunk
        target_chars = CHUNKING_MAX_TOKENS * AVG_CHARS_PER_TOKEN
        overlap_chars = CHUNKING_OVERLAP * AVG_CHARS_PER_TOKEN

        splits = []
        if is_code:
            # Code splitting: try splitting by 2+ newlines first
            splits = re.split(r'\n{2,}', text)
        else:
            # Text splitting: by paragraphs
            splits = re.split(r'\n{2,}', text)

        final_chunks = []
        current_chunk = []
        current_len = 0
        
        for split in splits:
            # Here we CLEAN the chunk content but keep it separate as a unit
            cleaned = clean_text(split)
            if not cleaned: continue
            
            # If a single split is massive (e.g. huge block of text without blank lines), force split it
            if len(cleaned) > target_chars + 100:
                # If current accumulation is non-empty, flush it first
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Slicing large block
                start = 0
                while start < len(cleaned):
                    end = start + target_chars
                    # If we can look ahead for a space, do so?
                    # Minimal implementation: hard slice
                    chunk_text = cleaned[start:end]
                    final_chunks.append(chunk_text)
                    start += (target_chars - overlap_chars)
                continue
            
            # Accumulate small chunks
            if current_len + len(cleaned) > target_chars:
                # Flush current
                joined = " ".join(current_chunk)
                if joined: final_chunks.append(joined)
                
                # Start new buffer. 
                # Ideally overlap with previous chunk's last sentences? 
                # For complexity, just clear buffer.
                current_chunk = [cleaned]
                current_len = len(cleaned)
            else:
                current_chunk.append(cleaned)
                current_len += len(cleaned) # +1 for space? roughly
        
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))
            
        return final_chunks
