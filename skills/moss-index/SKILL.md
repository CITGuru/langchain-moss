---
name: moss-index
description: Index documents into MOSS vector store. Ingest raw text, files (txt, pdf, md), or directories. Automatically chunks content using LangChain text splitters. Use when the user wants to add, store, or index documents, files, or text into the knowledge base.
---

# MOSS Document Indexing

Index documents into the MOSS vector store with automatic chunking and metadata support.

## Quick Start

```python
from langchain_moss import MossVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Initialize vector store
store = MossVectorStore(
    project_id=os.getenv("MOSS_PROJECT_ID"),
    project_key=os.getenv("MOSS_PROJECT_KEY"),
    index_name="my-documents"
)

# Index raw text with chunking
text = "Your document content here..."
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

ids = store.add_texts(
    texts=chunks,
    metadatas=[{"source": "user_input"} for _ in chunks]
)
```

## Indexing Options

### Option 1: Index Raw Text

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.
"""

# Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

# Index chunks
ids = store.add_texts(
    texts=chunks,
    metadatas=[{"source": "raw_text", "chunk_index": i} for i, _ in enumerate(chunks)]
)
print(f"Indexed {len(ids)} chunks")
```

### Option 2: Index Multiple Texts

```python
texts = [
    "Python is a programming language.",
    "Vector databases enable semantic search.",
    "RAG combines retrieval with generation."
]

ids = store.add_texts(
    texts=texts,
    metadatas=[
        {"topic": "programming"},
        {"topic": "databases"},
        {"topic": "AI"}
    ]
)
```

### Option 3: Index from File

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load text file
loader = TextLoader("document.txt")
documents = loader.load()

# Load PDF file
# loader = PyPDFLoader("document.pdf")
# documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Index
ids = store.add_documents(documents=chunks)
print(f"Indexed {len(ids)} chunks from file")
```

### Option 4: Index from Directory

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all .txt files from directory
loader = DirectoryLoader(
    "docs/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True
)
documents = loader.load()

# Split and index
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
ids = store.add_documents(documents=chunks)
```

## Chunking Strategies

### Recursive Character Splitter (Recommended)

Best for general text. Splits on multiple separators hierarchically.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
```

### Character Splitter

Simple splitting on a single separator.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n\n"
)
```

## Chunk Size Guidelines

| Content Type | Chunk Size | Overlap | Rationale |
|--------------|------------|---------|-----------|
| Short docs | 200-500 | 50 | Preserve precision |
| General text | 500-1000 | 100-200 | Balanced |
| Long articles | 1000-2000 | 200-300 | More context |
| Code | 500-1500 | 100 | Preserve functions |

## Adding Metadata

Always add metadata for filtering and tracking:

```python
ids = store.add_texts(
    texts=chunks,
    metadatas=[
        {
            "source": "filename.pdf",
            "page": 1,
            "topic": "machine_learning",
            "date_indexed": "2024-01-06",
            "author": "John Doe"
        }
        for _ in chunks
    ]
)
```

## Complete Indexing Workflow

```python
import os
from pathlib import Path
from langchain_moss import MossVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def index_documents(
    source: str,
    index_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict = None
):
    """Index documents from file, directory, or raw text."""
    
    # Initialize store
    store = MossVectorStore(
        project_id=os.getenv("MOSS_PROJECT_ID"),
        project_key=os.getenv("MOSS_PROJECT_KEY"),
        index_name=index_name
    )
    
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    path = Path(source)
    
    if path.exists():
        if path.is_dir():
            # Load from directory
            loader = DirectoryLoader(str(path), glob="**/*.*")
            documents = loader.load()
        elif path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            documents = loader.load()
        else:
            loader = TextLoader(str(path))
            documents = loader.load()
        
        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        # Split and index
        chunks = splitter.split_documents(documents)
        ids = store.add_documents(documents=chunks)
    else:
        # Treat as raw text
        chunks = splitter.split_text(source)
        ids = store.add_texts(
            texts=chunks,
            metadatas=[metadata or {} for _ in chunks]
        )
    
    return {
        "success": True,
        "num_chunks": len(ids),
        "document_ids": ids
    }

# Usage
result = index_documents(
    source="docs/guide.pdf",
    chunk_size=500,
    metadata={"project": "demo"}
)
print(f"Indexed {result['num_chunks']} chunks")
```

## Environment Variables

Required:
- `MOSS_PROJECT_ID` - Your MOSS project ID
- `MOSS_PROJECT_KEY` - Your MOSS project key

## Dependencies

```bash
pip install langchain-moss langchain-text-splitters langchain-community
```

