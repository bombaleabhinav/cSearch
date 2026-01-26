import os
from pathlib import Path
from typing import List, Dict, Generator
from PyPDF2 import PdfReader
from docx import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {".txt", ".md", ".py", ".js", ".html", ".css", ".c", ".cpp", ".h", ".pdf", ".docx"}

    def get_files(self, root_dir: str) -> List[Path]:
        """
        Recursively finds all supported files in the directory.
        """
        root_path = Path(root_dir)
        files = []
        for ext in self.supported_extensions:
            files.extend(root_path.rglob(f"*{ext}"))
        return files

    def read_file(self, file_path: Path) -> str:
        """
        Reads the content of a file, handling basic encoding issues and various formats.
        """
        try:
            ext = file_path.suffix.lower()
            if ext in {".txt", ".md", ".py", ".js", ".html", ".css", ".c", ".cpp", ".h"}:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
        return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunks text based on chunk_size and chunk_overlap.
        Attempts to split on double newlines (paragraphs) first, then single newlines, then spaces.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If we're not at the end, try to find a better breakpoint
            if end < text_len:
                # Look for paragraph break
                last_double_newline = text.rfind('\n\n', start, end)
                if last_double_newline != -1 and last_double_newline > start + (self.chunk_size // 2):
                    end = last_double_newline + 2
                else:
                    # Look for newline break
                    last_newline = text.rfind('\n', start, end)
                    if last_newline != -1 and last_newline > start + (self.chunk_size // 2):
                        end = last_newline + 1
                    else:
                        # Look for space
                        last_space = text.rfind(' ', start, end)
                        if last_space != -1 and last_space > start + (self.chunk_size // 2):
                            end = last_space + 1

            chunks.append(text[start:end].strip())
            
            # Move start forward by chunk_size - overlap, but ensure we progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start
                
        return [c for c in chunks if c]

    def process_directory(self, root_dir: str) -> Generator[Dict, None, None]:
        """
        Processes a directory and yields chunks with metadata.
        """
        files = self.get_files(root_dir)
        for file_path in files:
            content = self.read_file(file_path)
            if not content:
                continue
                
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                yield {
                    "content": chunk,
                    "metadata": {
                        "path": str(file_path),
                        "chunk_index": i,
                        "file_type": file_path.suffix
                    }
                }
