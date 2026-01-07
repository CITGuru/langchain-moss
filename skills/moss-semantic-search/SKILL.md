---
name: moss-semantic-search
description: Perform semantic similarity search on MOSS vector store. Find documents by meaning using embeddings. Supports top-k retrieval, score thresholds, and metadata filtering. Use when the user wants to search, find, query, or retrieve documents based on meaning or similarity.
---

# MOSS Semantic Search

Perform vector similarity search to find documents by semantic meaning.

## Quick Start

```python
from langchain_moss import MossVectorStore
import os

# Initialize vector store
store = MossVectorStore(
    project_id=os.getenv("MOSS_PROJECT_ID"),
    project_key=os.getenv("MOSS_PROJECT_KEY"),
    index_name="my-documents"
)

# Search
results = store.similarity_search(
    query="machine learning algorithms",
    k=5
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

## Search Methods

### Basic Similarity Search

Returns top-k most similar documents:

```python
results = store.similarity_search(
    query="artificial intelligence",
    k=5  # Number of results
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Search with Scores

Returns documents with similarity scores:

```python
results = store.similarity_search_with_score(
    query="neural networks",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print("---")
```

### Search with Score Threshold

Filter results by minimum similarity score:

```python
results = store.similarity_search_with_relevance_scores(
    query="deep learning",
    k=10,
    score_threshold=0.7  # Only return scores >= 0.7
)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content[:100]}...")
```

### Search with Metadata Filter

Filter results by metadata fields:

```python
results = store.similarity_search(
    query="programming concepts",
    k=5,
    filter={"topic": "python"}  # Only documents with topic="python"
)

# Multiple filters
results = store.similarity_search(
    query="API documentation",
    k=5,
    filter={
        "source": "docs",
        "version": "2.0"
    }
)
```

## Using as LangChain Retriever

Convert to retriever for use with LangChain chains:

```python
# Basic retriever
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

docs = retriever.invoke("machine learning")

# With score threshold
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.7
    }
)
```

## Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query text |
| `k` | int | 4 | Number of results to return |
| `filter` | dict | None | Metadata filter |
| `score_threshold` | float | None | Minimum similarity score (0-1) |

## Choosing k Value

| Use Case | Recommended k | Rationale |
|----------|--------------|-----------|
| Quick answer | 3-5 | Fast, focused results |
| Research | 10-20 | More coverage |
| Comprehensive | 20-50 | Thorough search |
| Re-ranking | 50-100 | Get candidates for re-ranking |

## Complete Search Function

```python
import os
from typing import List, Dict, Any, Optional
from langchain_moss import MossVectorStore

def semantic_search(
    query: str,
    index_name: str = "documents",
    k: int = 5,
    score_threshold: Optional[float] = None,
    filter: Optional[Dict[str, Any]] = None,
    return_scores: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on MOSS vector store.
    
    Args:
        query: Search query text
        index_name: Name of the MOSS index
        k: Number of results to return
        score_threshold: Minimum similarity score (0-1)
        filter: Metadata filter dictionary
        return_scores: Whether to include scores
        
    Returns:
        List of search results with content, metadata, and optional scores
    """
    store = MossVectorStore(
        project_id=os.getenv("MOSS_PROJECT_ID"),
        project_key=os.getenv("MOSS_PROJECT_KEY"),
        index_name=index_name
    )
    
    if return_scores:
        results = store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            score_threshold=score_threshold
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "id": doc.id
            }
            for doc, score in results
        ]
    else:
        results = store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "id": doc.id
            }
            for doc in results
        ]

# Usage examples
results = semantic_search(
    query="machine learning algorithms",
    k=5,
    score_threshold=0.6
)

for r in results:
    print(f"[{r['score']:.3f}] {r['content'][:100]}...")

# With metadata filter
results = semantic_search(
    query="API endpoints",
    k=10,
    filter={"doc_type": "api_reference"}
)
```

## Integration with RAG

Use semantic search in a Retrieval-Augmented Generation pipeline:

```python
from langchain_moss import MossVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize
store = MossVectorStore(
    project_id=os.getenv("MOSS_PROJECT_ID"),
    project_key=os.getenv("MOSS_PROJECT_KEY"),
    index_name="knowledge_base"
)

retriever = store.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4")

# Create RAG chain
template = """Answer based on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query
answer = rag_chain.invoke("What is machine learning?")
print(answer)
```

## Tips for Better Results

1. **Use natural language queries**: "How does authentication work?" > "authentication"
2. **Be specific**: "Python async error handling" > "errors"
3. **Match document language**: Query in the same language as indexed docs
4. **Adjust k based on need**: Start with k=10, tune from there
5. **Use metadata filters**: Narrow search space for faster, more relevant results

## Environment Variables

Required:
- `MOSS_PROJECT_ID` - Your MOSS project ID
- `MOSS_PROJECT_KEY` - Your MOSS project key

## Dependencies

```bash
pip install langchain-moss
```

