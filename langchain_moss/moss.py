from __future__ import annotations

import asyncio
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from inferedge_moss import MossClient, DocumentInfo, AddDocumentsOptions, QueryResultDocumentInfo


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence



class MossModel(str, Enum):
    MOSS_MINILM = "moss-minilm"
    MOSS_MEDIUMLM = "moss-mediumlm"

class MossVectorStoreError(Exception):
    """`MossVectorStore` related exceptions."""


class MossVectorStore(VectorStore):
    """Moss vector store integration.

    Setup:
        Install `langchain-moss` package.

        ```bash
        pip install -qU langchain-moss
        ```

    Key init args:
        index_name:
            Name of the index/collection.
        client:
            Optional Moss client to use. If not provided, project_id and project_key must be provided.
        project_id:
            Optional Moss project ID. Required if client is not provided.
        project_key:
            Optional Moss project key. Required if client is not provided.

    Instantiate:
        ```python
        from langchain_moss import MossVectorStore
        from inferedge_moss import MossClient

        client = MossClient(project_id="your-project-id", project_key="your-project-key")

        vector_store = MossVectorStore(
            client=client,
            index_name="demo_collection",
        )
        ```

        Or with project credentials:

        ```python
        vector_store = MossVectorStore(
            project_id="your-project-id",
            project_key="your-project-key",
            index_name="demo_collection",
        )
        ```

    Add Documents:
        ```python
        from langchain_core.documents import Document
        from uuid import uuid4

        document_1 = Document(page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(page_content="thud", metadata={"bar": "baz"})
        document_3 = Document(page_content="i will be deleted :(")

        documents = [document_1, document_2, document_3]
        ids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)
        ```

    Delete Documents:
        ```python
        vector_store.delete(ids=[ids[-1]])
        ```

    Search:
        ```python
        results = vector_store.similarity_search(
            query="thud",
            k=1,
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        *thud[
            {
                "bar": "baz",
                "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                "_index_name": "demo_collection",
            }
        ]
        ```

    Search with filter:
        ```python
        results = vector_store.similarity_search(
            query="thud",
            k=1,
            filter={"metadata.bar": "baz"},
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        *thud[
            {
                "bar": "baz",
                "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                "_index_name": "demo_collection",
            }
        ]
        ```

    Search with score:
        ```python
        results = vector_store.similarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.832268] foo [{'baz': 'bar', '_id': '44ec7094-b061-45ac-8fbf-014b0f18e8aa', '_index_name': 'demo_collection'}]
        ```

    Async:
        ```python
        # add documents
        # await vector_store.aadd_documents(documents=documents, ids=ids)

        # delete documents
        # await vector_store.adelete(ids=["3"])

        # search
        # results = vector_store.asimilarity_search(query="thud",k=1)

        # search with score
        results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.832268] foo [{'baz': 'bar', '_id': '44ec7094-b061-45ac-8fbf-014b0f18e8aa', '_index_name': 'demo_collection'}]
        ```

    Use as Retriever:
        ```python
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.7},
        )
        retriever.invoke("thud")
        ```

        ```python
        [
            Document(
                metadata={
                    "bar": "baz",
                    "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                    "_index_name": "demo_collection",
                },
                page_content="thud",
            )
        ]
        ```
    """  # noqa: E501

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        index_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        client: MossClient | None = None,
        project_id: str | None = None,
        project_key: str | None = None,
        model_id: Union[MossModel, str] = MossModel.MOSS_MINILM,
    ) -> None:
        """Initialize a new instance of `MossVectorStore`.

        Args:
            index_name: Name of the index/collection to use.
            client: Optional Moss client instance. If not provided, project_id and project_key must be provided.
            project_id: Optional Moss project ID. Required if client is not provided.
            project_key: Optional Moss project key. Required if client is not provided.

        Raises:
            ValueError: If neither client nor both project_id and project_key are provided.

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from inferedge_moss import MossClient

            client = MossClient(project_id="your-project-id", project_key="your-project-key")
            vector_store = MossVectorStore(
                client=client,
                index_name="my-collection",
            )
            ```
        """

        if client is None:
            if project_id is None or project_key is None:
                raise ValueError("Either 'client' or both 'project_id' and 'project_key' must be provided.")
            client = MossClient(project_id, project_key)

        self._client = client
        self.index_name = index_name
        self.index_loaded = False

        # Check if index exists, create if it doesn't
        try:
            asyncio.run(self._client.get_index(index_name))
        except Exception:
            # Index doesn't exist, create it
            self._create_index(index_name, model_id)

        # Load the index to ensure it's ready for queries
        self._load_index(index_name)
        self.index_loaded = True

    def _create_index(self, index_name: str, model_id: Union[MossModel, str]) -> None:
        if isinstance(model_id, MossModel):
            model_id = model_id.value
        return asyncio.run(self._client.create_index(index_name, [], model_id))

    def _load_index(self, index_name: str) -> None:
        return asyncio.run(self._client.load_index(index_name))

    @property
    def client(self) -> MossClient:
        """Get the Moss client instance that is being used.

        Returns:
            MossClient: An instance of `MossClient`.

        """
        return self._client

    @property
    def embeddings(self) -> Embeddings | None:
        """Get the dense embeddings instance that is being used.

        Returns:
            Embeddings: An instance of `Embeddings`.

        """
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Texts to add to the `VectorStore`.
            metadatas: Optional list of metadatas.
                    When querying, you can filter on this metadata.
            ids: Optional list of IDs. (Items without IDs will be assigned UUIDs)
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: When metadata is incorrect.
        """
        texts = list(texts)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

        documents = []

        for _id, text, metadata in zip(
            ids,
            texts,
            metadatas or [{}] * len(texts),
        ):
            documents.append(
                DocumentInfo(
                    id=_id,
                    text=text,
                    metadata=metadata or {},
                )
            )

        options = AddDocumentsOptions(
            upsert=True,
        )

        asyncio.run(
            self._client.add_docs(
                index_name=self.index_name,
                docs=documents,
                options=options,
            )
        )
        if documents:
            self.index_loaded = False

        return ids
    

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any | None = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        index_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        client: MossClient | None = None,
        project_id: str | None = None,
        project_key: str | None = None,
        **kwargs: Any,
    ) -> MossVectorStore:
        """Create a `MossVectorStore` from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Not used in Moss.
            metadatas: Optional list of metadata dicts associated with the texts.
            ids: Optional list of ids to associate with the documents.
            index_name: Name of the index to work with.
            client: Optional Moss client instance.
            project_id: Optional Moss project ID.
            project_key: Optional Moss project key.
            **kwargs: Additional keyword arguments.

        Returns:
            MossVectorStore: A new `MossVectorStore` instance with the texts added.

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from inferedge_moss import MossClient

            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Hello world",
                "Machine learning is fascinating"
            ]
            metadatas = [
                {"source": "book", "page": 1},
                {"source": "greeting", "language": "english"},
                {"source": "article", "topic": "AI"}
            ]

            client = MossClient(project_id="your-project-id", project_key="your-project-key")

            vector_store = MossVectorStore.from_texts(
                texts=texts,
                client=client,
                metadatas=metadatas,
                index_name="langchain-demo",
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("AI and machine learning", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method creates a new `MossVectorStore` instance and adds the
                provided texts to it.
            - If `metadatas` is provided, it must have the same length as `texts`.
            - If `ids` is provided, it must have the same length as `texts`.
        """

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        args = {}
        if client is not None:
            args['client'] = client
        elif project_id is not None and project_key is not None:
            args['project_id'] = project_id
            args['project_key'] = project_key

        vector_store = cls(index_name=index_name, **args)
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return vector_store


    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Any | None = None,
        index_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        client: MossClient | None = None,
        project_id: str | None = None,
        project_key: str | None = None,
        **kwargs: Any,
    ) -> MossVectorStore:
        """Create a `MossVectorStore` from a list of `Document` objects.

        Args:
            documents: List of `Document` objects to add to the vector store.
            embedding: Not used in Moss.
            index_name: Name of the index to work with.
            client: Optional Moss client instance.
            project_id: Optional Moss project ID.
            project_key: Optional Moss project key.
            **kwargs: Additional keyword arguments.

        Returns:
            MossVectorStore: A new `MossVectorStore` instance with the documents
                added.

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from langchain_core.documents import Document
            from inferedge_moss import MossClient

            documents = [
                Document(
                    page_content="The quick brown fox",
                    metadata={"animal": "fox"}
                ),
                Document(
                    page_content="jumps over the lazy dog",
                    metadata={"animal": "dog"}
                )
            ]

            client = MossClient(project_id="your-project-id", project_key="your-project-key")

            vector_store = MossVectorStore.from_documents(
                documents=documents,
                client=client,
                index_name="animal-docs",
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("quick animal", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method creates a new `MossVectorStore` instance and adds the
                provided documents to it.
            - The method extracts the text content and metadata from
                each `Document` object.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id if hasattr(doc, 'id') and doc.id is not None else str(uuid.uuid4()) for doc in documents]

        return cls.from_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
            index_name=index_name,
            client=client,
            project_id=project_id,
            project_key=project_key,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        client: MossClient | None = None,
        project_id: str | None = None,
        project_key: str | None = None,
        **kwargs: Any,
    ) -> MossVectorStore:
        """Create a `MossVectorStore` from an existing Moss index.

        This method allows you to connect to an already existing index in Moss,
        which can be useful for continuing work with previously created indexes
        or for connecting to indexes created outside of this client.

        Args:
            index_name: Name of the existing index to use.
            client: Optional Moss client instance. If not provided, project_id and project_key must be provided.
            project_id: Optional Moss project ID. Required if client is not provided.
            project_key: Optional Moss project key. Required if client is not provided.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            MossVectorStore: A new `MossVectorStore` instance connected to the
                existing index.

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from inferedge_moss import MossClient

            client = MossClient(project_id="your-project-id", project_key="your-project-key")

            # Connect to an existing index
            vector_store = MossVectorStore.from_existing_index(
                index_name="my-existing-index",
                client=client,
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("AI and machine learning", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method assumes that the index already exists in Moss.
            - This method is useful for scenarios where you want to reuse an
                existing index, such as when the index was created by another process
                or when you want to use the same index across different sessions
                or applications.
        """
        return MossVectorStore(
            index_name=index_name,
            client=client,
            project_id=project_id,
            project_key=project_key,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete ids from the vector store.

        Args:
            ids: Optional list of ids of the documents to delete.
            **kwargs: Additional keyword arguments (not used in the
                current implementation).

        Returns:
            Optional[bool]: `True` if one or more keys are deleted, `False` otherwise

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from inferedge_moss import MossClient

            client = MossClient(project_id="your-project-id", project_key="your-project-key")
            vector_store = MossVectorStore(
                index_name="langchain-demo",
                client=client,
            )

            # Assuming documents with these ids exist in the store
            ids_to_delete = ["doc1", "doc2", "doc3"]

            result = vector_store.delete(ids=ids_to_delete)
            if result:
                print("Documents were successfully deleted")
            else:
                print("No Documents were deleted")
            ```

        Note:
            - If `ids` is `None` or an empty list, the method returns `False`.
            - The method uses the Moss client to delete documents by their IDs.
        """
        if ids and len(ids) > 0:
            try:
                # Assuming MossClient has a delete_docs method
                # If the API differs, this will need to be adjusted
                asyncio.run(self._client.delete_docs(self.index_name, ids))
                return True
            except Exception:
                return False
        else:
            return False

    def _apply_filter(self, docs: List[DocumentInfo], _filter: Optional[Dict[str, Any]]) -> List[DocumentInfo]:
        if _filter:
            filtered_docs = []
            for doc in docs:
                matches = True
                for key, value in _filter.items():
                    metadata_key = key.replace("metadata.", "") if key.startswith("metadata.") else key
                    if doc.metadata.get(metadata_key) != value:
                        matches = False
                        break
                if matches:
                    filtered_docs.append(doc)
            return filtered_docs
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of `Document` objects most similar to the query.
        """
        results_with_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in results_with_scores]

    def _build_document_from_result(self, res: DocumentInfo) -> Document:
        """Build a `Document` object from a Moss query result."""
        content = res.text
        metadata = res.metadata or {}

        return Document(page_content=content, metadata=metadata, id=res.id)
    
    def _build_document_with_score_from_result(
        self, res: QueryResultDocumentInfo
    ) -> Tuple[Document, float]:
        """Build a `Document` object with score from a Moss query result."""
        doc = self._build_document_from_result(res)
        score = res.score
        return (doc, float(score))
    
    def _transform_documents(
        self,
        results: List[DocumentInfo],
    ) -> List[Document]:
        docs = [self._build_document_from_result(res) for res in results]
        return docs
    
    def _transform_documents_with_scores(
        self,
        results: List[QueryResultDocumentInfo],
    ) -> List[Tuple[Document, float]]:
        docs = [self._build_document_with_score_from_result(res) for res in results]
        return docs

    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        r"""Return documents most similar to query string, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function.

                Common kwargs include:
                - `score_threshold`: Optional distance threshold for filtering
               

        Returns:
            List of tuples of `(Document, score)` most similar to the query.

        Example:
            ```python
            from langchain_moss import MossVectorStore
            from inferedge_moss import MossClient

            client = MossClient(project_id="your-project-id", project_key="your-project-key")
            vector_store = MossVectorStore(
                index_name="langchain-demo",
                client=client,
            )

            results = vector_store.similarity_search_with_score(
                "What is machine learning?",
                k=2,
                filter=None
            )

            for doc, score in results:
                print(f"Score: {score}")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}\n")
            ```

        Note:
            - The method returns scores along with documents. The score interpretation
                depends on Moss's scoring mechanism.
            - The actual search is performed directly using the query string with Moss.
            - The `filter` parameter allows for additional filtering of results
                based on metadata (currently not implemented).
        """
        score_threshold = kwargs.get("score_threshold", None)

        if not self.index_loaded:
            self._load_index(self.index_name)
            self.index_loaded = True

        results = asyncio.run(
            self.client.query(self.index_name, query, top_k=k, alpha=score_threshold)
        )

        # Apply metadata filter if provided. NOTE: this is temporary until Moss supports metadata filtering.
        if filter:
            results.docs = self._apply_filter(results.docs, filter)

        docs = self._transform_documents_with_scores(results.docs)
        return docs

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query string, along with relevance scores.

        This method requires a score_threshold to be provided in kwargs.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function.
                Must include `score_threshold` (required).

        Returns:
            List of tuples of `(Document, score)` most similar to the query.

        Raises:
            ValueError: If `score_threshold` is not provided in kwargs.
        """
        if "score_threshold" not in kwargs:
            raise ValueError("score_threshold is required in kwargs for similarity_search_with_relevance_scores")
        
        return self.similarity_search_with_score(query, k, filter, **kwargs)

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of `Document` objects.

        """

        docs = asyncio.run(self.client.get_docs(self.index_name, {"doc_ids": ids}))

        documents = [self._build_document_from_result(doc) for doc in docs]

        return documents

