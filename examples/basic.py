import os

from inferedge_moss import MossClient

from langchain_moss import MossVectorStore


def main() -> None:
    client = MossClient(project_id=os.getenv("MOSS_PROJECT_ID"), project_key=os.getenv("MOSS_PROJECT_KEY"))
    index_name = os.getenv("MOSS_INDEX_NAME", "langchain")

    store = MossVectorStore(client=client, index_name=index_name)

    print(store.index_name)
    print(store.index_loaded)


if __name__ == "__main__":
    main()
