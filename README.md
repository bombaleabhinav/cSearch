## cSearch

The name **cSearch** comes from the four core principles behind the system:

- **Context** — understands the surrounding meaning of text instead of relying on exact keyword matches.
- **Cognitive** — models user intent and semantic relationships when retrieving files.
- **Conceptual** — finds files based on concepts and ideas, not just literal words.
- **Content** — searches inside file contents across multiple formats, not just filenames or metadata.

Together, these make cSearch a *meaning-first* file retrieval system.

## Features
- **Pure Semantic Search**: Uses vector embeddings (`nomic-embed-text`) to find relevant files based on content meaning, not filenames.
- **Local & Private**: No cloud APIs, no data leaves your machine.
- **Multi-format Support**: Indexes `.txt`, `.md`, `.py`, `.js`, `.pdf`, `.docx`, and more.
- **Intelligent Chunking**: Splits large files into meaningful segments with overlap for better context retrieval.
- **Fast Retrieval**: Powered by FAISS (Facebook AI Similarity Search) for near-instant search once indexed.
