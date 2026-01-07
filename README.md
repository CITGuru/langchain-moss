# langchain-moss

LangChain integration for [Moss](https://usemoss.dev/) vector database. This package provides a `MossVectorStore` that implements LangChain's `VectorStore` interface, enabling seamless integration with LangChain applications.

## Installation

```bash
pip install langchain-moss
```

or with local installation:

```bash
git clone https://github.com/citguru/langchain-moss.git
cd langchain-moss
pip install -e .
```

## Quick Start

### 1. Initialize the Vector Store

```python
from langchain_moss import MossVectorStore, MossModel
from inferedge_moss import MossClient

# Create a Moss client
client = MossClient(
    project_id="your-project-id",
    project_key="your-project-key"
)

# Initialize the vector store with a Moss client
vector_store = MossVectorStore(
    client=client,
    index_name="my-index",
    # model_id=MossModel.MOSS_MINILM (default) or MossModel.MOSS_MEDIUMLM
)
```

Or initialize directly with project credentials:

```python
vector_store = MossVectorStore(
    project_id="your-project-id",
    project_key="your-project-key",
    index_name="my-index",
    # model_id=MossModel.MOSS_MINILM (default) or MossModel.MOSS_MEDIUMLM
)
```

### 2. Add Documents

```python
# Add texts with metadata
ids = vector_store.add_texts(
    texts=[
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Vector databases enable semantic search"
    ],
    metadatas=[
        {"topic": "AI", "source": "docs"},
        {"topic": "programming", "source": "docs"},
        {"topic": "databases", "source": "docs"}
    ]
)
```

You can also add `Document` objects directly:

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence",
        metadata={"topic": "AI", "source": "docs"}
    ),
    Document(
        page_content="Python is a popular programming language",
        metadata={"topic": "programming", "source": "docs"}
    )
]

ids = vector_store.add_documents(documents=documents)
```

### 3. Search

```python
# Similarity search
results = vector_store.similarity_search(
    query="artificial intelligence",
    k=2
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)

# Search with scores
results_with_scores = vector_store.similarity_search_with_score(
    query="programming languages",
    k=2
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(doc.page_content)

# Search with metadata filter
filtered_results = vector_store.similarity_search(
    query="programming",
    k=2,
    filter={"topic": "programming"}
)

# Search with relevance scores (requires score_threshold)
results_with_relevance = vector_store.similarity_search_with_relevance_scores(
    query="machine learning",
    k=2,
    score_threshold=0.7
)
```

### 4. Retrieve Documents by ID

```python
# Get documents by their IDs
documents = vector_store.get_by_ids(ids=["doc-id-1", "doc-id-2"])
for doc in documents:
    print(f"ID: {doc.id}")
    print(f"Content: {doc.page_content}")
```

### 5. Delete Documents

```python
# Delete documents by their IDs
vector_store.delete(ids=["doc-id-1", "doc-id-2"])
```

### 6. Use as LangChain Retriever

```python
# Similarity search retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke("machine learning")

# Similarity score threshold retriever
score_threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.7}
)

docs = score_threshold_retriever.invoke("machine learning")
```

### 7. Async Operations

All operations support async versions:

```python
import asyncio

# Async add documents
ids = await vector_store.aadd_documents(documents=documents)

# Async search
results = await vector_store.asimilarity_search(query="AI", k=2)

# Async search with scores
results = await vector_store.asimilarity_search_with_score(query="AI", k=2)

# Async delete
await vector_store.adelete(ids=["doc-id-1"])
```

## Key Features

- **Automatic Index Management**: Automatically creates and loads indexes if they don't exist
- **Full LangChain Integration**: Implements the `VectorStore` interface for seamless use with LangChain
- **Metadata Support**: Filter results by metadata
- **Score-based Search**: Get similarity scores with search results
- **Document Management**: Add, retrieve, and delete documents by ID
- **Async Support**: All operations have async counterparts

## API Overview

### Initialization

```python
MossVectorStore(
    index_name: str = "langchain",
    client: MossClient | None = None,
    project_id: str | None = None,
    project_key: str | None = None,
    model_id: MossModel | str = MossModel.MOSS_MINILM
)
```

**Parameters:**
- `index_name`: Name of the index/collection to use (default: "langchain")
- `client`: Optional Moss client instance. If not provided, `project_id` and `project_key` must be provided
- `project_id`: Optional Moss project ID. Required if `client` is not provided
- `project_key`: Optional Moss project key. Required if `client` is not provided
- `model_id`: Embedding model to use. Options: `MossModel.MOSS_MINILM` (default) or `MossModel.MOSS_MEDIUMLM`

### Key Methods

#### Document Management

- `add_texts(texts, metadatas=None, ids=None)` - Add texts to the vector store
- `add_documents(documents, ids=None)` - Add Document objects to the vector store
- `get_by_ids(ids)` - Retrieve documents by their IDs
- `delete(ids)` - Delete documents by their IDs

#### Search Methods

- `similarity_search(query, k=4, filter=None)` - Search for similar documents
- `similarity_search_with_score(query, k=4, filter=None, score_threshold=None)` - Search with similarity scores
- `similarity_search_with_relevance_scores(query, k=4, filter=None, score_threshold)` - Search with relevance scores (requires `score_threshold`)

#### Retriever

- `as_retriever(**kwargs)` - Get a LangChain retriever
  - `search_type`: "similarity" or "mmr"
  - `search_kwargs`: Dictionary of search parameters (e.g., `{"k": 3}`, `{"k": 3, "fetch_k": 5, "lambda_mult": 0.5}` for MMR)

#### Async Methods

- `aadd_texts(texts, metadatas=None, ids=None)` - Async version of `add_texts`
- `aadd_documents(documents, ids=None)` - Async version of `add_documents`
- `asimilarity_search(query, k=4, filter=None)` - Async version of `similarity_search`
- `asimilarity_search_with_score(query, k=4, filter=None, score_threshold=None)` - Async version of `similarity_search_with_score`
- `adelete(ids)` - Async version of `delete`

### Class Methods

- `from_texts(texts, metadatas=None, ids=None, ...)` - Create vector store from texts
- `from_documents(documents, ...)` - Create vector store from Document objects
- `from_existing_index(index_name, ...)` - Connect to an existing index

### Example: Using Class Methods

```python
# Create from texts
vector_store = MossVectorStore.from_texts(
    texts=["Text 1", "Text 2", "Text 3"],
    metadatas=[{"source": "1"}, {"source": "2"}, {"source": "3"}],
    project_id="your-project-id",
    project_key="your-project-key",
    index_name="my-index"
)

# Create from documents
from langchain_core.documents import Document

documents = [
    Document(page_content="Text 1", metadata={"source": "1"}),
    Document(page_content="Text 2", metadata={"source": "2"})
]

vector_store = MossVectorStore.from_documents(
    documents=documents,
    project_id="your-project-id",
    project_key="your-project-key",
    index_name="my-index"
)

# Connect to existing index
vector_store = MossVectorStore.from_existing_index(
    index_name="existing-index",
    project_id="your-project-id",
    project_key="your-project-key"
)
```

## More Examples

See the [examples directory](examples/README.md) for complete working examples:

- `examples/basic.py` - Basic initialization
- `examples/add_and_search.py` - Adding documents and searching
- `examples/retriever.py` - Using as a LangChain retriever

To run the examples, set the following environment variables:

```bash
export MOSS_PROJECT_ID="your-project-id"
export MOSS_PROJECT_KEY="your-project-key"
export MOSS_INDEX_NAME="langchain"  # optional
```

Then run:

```bash
python examples/basic.py
python examples/add_and_search.py
python examples/retriever.py
```

## Requirements

- Python >= 3.13
- `inferedge-moss >= 1.0.0b8`
- `langchain-core >= 1.2.6`
- `pydantic >= 2.12.5`
- `asyncio >= 4.0.0`

## License

See LICENSE file for details.