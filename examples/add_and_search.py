import os

from inferedge_moss import MossClient

from langchain_moss import MossVectorStore




def main() -> None:
    client = MossClient(project_id=os.getenv("MOSS_PROJECT_ID"), project_key=os.getenv("MOSS_PROJECT_KEY"))
    index_name = os.getenv("MOSS_INDEX_NAME", "langchain")

    store = MossVectorStore(client=client, index_name=index_name)

    ids = store.add_texts(
        texts=[
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "Vector databases enable semantic search and similarity matching",
        ],
        metadatas=[
            {"source": "examples/add_and_search", "topic": "animals"},
            {"source": "examples/add_and_search", "topic": "AI"},
            {"source": "examples/add_and_search", "topic": "programming"},
            {"source": "examples/add_and_search", "topic": "databases"},
        ],
    )

    docs = store.similarity_search("artificial intelligence", k=2)
    for d in docs:
        print(d.page_content)

    scored = store.similarity_search_with_score("programming languages", k=2)
    for d, score in scored:
        print(score, d.page_content)

    got = store.get_by_ids(ids[:2])
    for d in got:
        print(d.id, d.page_content)

    store.delete(ids=[ids[-1]])


if __name__ == "__main__":
    main()
