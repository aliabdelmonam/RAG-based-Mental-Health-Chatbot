from src.db.vector_db_interface import VectorDBInterface


class ChromaProvider(VectorDBInterface):
    """
    Placeholder/Stub for Chroma DB Provider.
    """

    def connect(self) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def disconnect(self) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def create_collection(
        self, collection_name: str, dimension: int, metric: str = "cosine"
    ) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def delete_collection(self, collection_name: str) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def upsert(self, collection_name: str, records: list) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def delete(self, collection_name: str, ids: list) -> None:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 5,
        filters: dict = None,
        include_vectors: bool = False,
    ) -> list:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def get_by_ids(self, collection_name: str, ids: list) -> list:
        raise NotImplementedError("ChromaProvider is not implemented yet.")

    def count(self, collection_name: str) -> int:
        raise NotImplementedError("ChromaProvider is not implemented yet.")
