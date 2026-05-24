from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    """
    Represents a vector record to be stored or retrieved from the vector database.
    """
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    Represents the output structure of a similarity search query.
    """
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


class VectorDBInterface(ABC):
    """
    Abstract Base Class that serves as the interface for any Vector Database provider.
    Inherit from this class to integrate databases like Chroma, Pinecone, Qdrant, etc.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the vector database client/service.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the vector database client/service.
        """
        pass

    @abstractmethod
    def create_collection(
        self, collection_name: str, dimension: int, metric: str = "cosine"
    ) -> None:
        """
        Create a collection (or index) with a given name, vector dimension, and distance metric.

        Args:
            collection_name: The name of the collection/index to create.
            dimension: The size of the embedding vectors.
            metric: The similarity metric to use (e.g., 'cosine', 'l2', 'ip').
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection (or index) and all its records.

        Args:
            collection_name: The name of the collection/index to delete.
        """
        pass

    @abstractmethod
    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        """
        Insert or update vector records in the specified collection.

        Args:
            collection_name: The name of the collection/index.
            records: A list of VectorRecord objects containing ID, vector, and payload.
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete specific records by their IDs from the collection.

        Args:
            collection_name: The name of the collection/index.
            ids: A list of record IDs to delete.
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """
        Perform a similarity search in the collection using the query vector.

        Args:
            collection_name: The name of the collection/index.
            query_vector: The embedding vector to query against.
            limit: The maximum number of search results to return.
            filters: Optional dictionary of metadata filters to apply.
            include_vectors: Whether to include the embedding vectors in the results.

        Returns:
            A list of SearchResult objects matching the query.
        """
        pass

    @abstractmethod
    def get_by_ids(self, collection_name: str, ids: List[str]) -> List[VectorRecord]:
        """
        Retrieve specific records by their IDs.

        Args:
            collection_name: The name of the collection/index.
            ids: A list of record IDs to retrieve.

        Returns:
            A list of VectorRecord objects found matching the IDs.
        """
        pass

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """
        Return the total number of vector records in the collection.

        Args:
            collection_name: The name of the collection/index.

        Returns:
            The count of records.
        """
        pass
