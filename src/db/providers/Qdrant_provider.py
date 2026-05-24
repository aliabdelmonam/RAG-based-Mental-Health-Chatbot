import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure src is in python path to allow clean relative imports
src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.logger import get_logger
from db.VectorDBENUM import QDrantVectorDB
from db.vector_db_interface import (
    SearchResult,
    VectorDBInterface,
    VectorRecord,
)

logger = get_logger("QDrantProvider")


class QDrantProvider(VectorDBInterface):
    """
    Qdrant implementation of the VectorDBInterface.
    """

    def __init__(
        self,
        location: Optional[Union[str, QDrantVectorDB]] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
    ):
        # Resolve location from arguments or environment
        raw_location = location or os.getenv("QDRANT_LOCATION")
        if isinstance(raw_location, QDrantVectorDB):
            raw_location = raw_location.value

        self.url = url or os.getenv("QDRANT_URL")
        self.host = host or os.getenv("QDRANT_HOST")
        self.port = (
            port
            or (int(os.getenv("QDRANT_PORT")) if os.getenv("QDRANT_PORT") else None)
        )
        self.path = path or os.getenv("QDRANT_PATH")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")

        # Map enum or string keys to proper Qdrant Client configuration
        if raw_location == "in_memory" or raw_location == ":memory:":
            self.location = ":memory:"
        elif raw_location == "persistent":
            self.location = None
            # Default local path for persistence if not provided
            self.path = self.path or os.path.join(
                os.getcwd(), "data", "qdrant_db"
            )
        else:
            self.location = raw_location

        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.prefix = prefix
        self.timeout = timeout

        # Fallback default: if no url, host, path, or location is defined, use in-memory
        if not any([self.location, self.url, self.host, self.path]):
            self.location = ":memory:"

        logger.info(
            "Initializing QDrantProvider: location=%s, url=%s, host=%s, port=%s, path=%s",
            self.location,
            self.url,
            self.host,
            self.port,
            self.path,
        )
        self.client = None

    def connect(self) -> None:
        """
        Establish connection to Qdrant.
        """
        if self.client is not None:
            return

        logger.info("Connecting to Qdrant...")
        from qdrant_client import QdrantClient

        try:
            if self.location is not None:
                self.client = QdrantClient(
                    location=self.location,
                    url=self.url,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    https=self.https,
                    api_key=self.api_key,
                    prefix=self.prefix,
                    timeout=self.timeout,
                    host=self.host,
                    path=self.path,
                )
            else:
                self.client = QdrantClient(
                    url=self.url,
                    host=self.host,
                    port=self.port,
                    path=self.path,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            logger.error("Failed to connect to Qdrant client: %s", e)
            raise

    def disconnect(self) -> None:
        """
        Close the connection to Qdrant.
        """
        if self.client is not None:
            logger.info("Disconnecting from Qdrant...")
            try:
                self.client.close()
                logger.info("Qdrant connection closed successfully.")
            except Exception as e:
                logger.warning("Error closing Qdrant client connection: %s", e)
            self.client = None

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: str = 'cosine',
    ) -> None:
        """
        Create a Qdrant collection if it does not exist.
        """
        if self.client is None:
            logger.error("Attempted create_collection without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        from qdrant_client.http import models as qmodels

        # distance_metric = self._map_metric(metric)
        if metric == 'cosine':
            distance_metric = QDrantVectorDB.COSINE.value
        elif metric == 'euclidean':
            distance_metric = QDrantVectorDB.EUCLIDEAN.value
        elif metric == 'dot_product':
            distance_metric = QDrantVectorDB.DOT_PRODUCT.value
        elif metric == 'manhattan':
            distance_metric = QDrantVectorDB.MANHATTAN.value
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

        logger.info(
            "Checking if Qdrant collection '%s' exists...", collection_name
        )
        exists = False
        try:
            if hasattr(self.client, "collection_exists"):
                exists = self.client.collection_exists(collection_name)
            else:
                self.client.get_collection(collection_name)
                exists = True
        except Exception:
            exists = False

        if not exists:
            logger.info(
                "Creating collection '%s' with dimension=%d, metric=%s",
                collection_name,
                dimension,
                metric,
            )
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=dimension, distance=distance_metric
                    ),
                )
                logger.info("Collection '%s' created successfully.", collection_name)
            except Exception as e:
                logger.error(
                    "Failed to create collection '%s': %s", collection_name, e
                )
                raise
        else:
            logger.info("Collection '%s' already exists.", collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from Qdrant.
        """
        if self.client is None:
            logger.error("Attempted delete_collection without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        logger.warning("Deleting collection '%s' from Qdrant.", collection_name)
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info("Collection '%s' deleted successfully.", collection_name)
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", collection_name, e)
            raise

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        """
        Insert or update vector records in Qdrant.
        """
        if self.client is None:
            logger.error("Attempted upsert without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        from qdrant_client.http import models as qmodels

        logger.info(
            "Upserting %d records into collection '%s'...",
            len(records),
            collection_name,
        )
        points = []
        for record in records:
            qdrant_id = self._to_qdrant_id(record.id)
            payload = record.payload.copy() if record.payload else {}
            # Store original ID to allow reverse mapping
            payload["_original_id"] = record.id

            points.append(
                qmodels.PointStruct(
                    id=qdrant_id, vector=record.vector, payload=payload
                )
            )

        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(
                "Successfully upserted %d records into '%s'.",
                len(records),
                collection_name,
            )
        except Exception as e:
            logger.error(
                "Failed to upsert records into collection '%s': %s",
                collection_name,
                e,
            )
            raise

    def delete(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete specific records by IDs from Qdrant.
        """
        if self.client is None:
            logger.error("Attempted delete without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        from qdrant_client.http import models as qmodels

        logger.info(
            "Deleting %d records from collection '%s'...",
            len(ids),
            collection_name,
        )
        qdrant_ids = [self._to_qdrant_id(rid) for rid in ids]

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=qdrant_ids),
            )
            logger.info(
                "Successfully deleted records from '%s'.", collection_name
            )
        except Exception as e:
            logger.error(
                "Failed to delete records from collection '%s': %s",
                collection_name,
                e,
            )
            raise

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """
        Perform a similarity search in Qdrant.
        """
        if self.client is None:
            logger.error("Attempted search without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        logger.debug(
            "Performing search in '%s' with limit=%d, filters=%s",
            collection_name,
            limit,
            filters,
        )
        q_filter = self._map_filters(filters) if filters else None

        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=q_filter,
                with_payload=True,
                with_vectors=include_vectors,
            )
            logger.debug(
                "Search in '%s' found %d results.",
                collection_name,
                len(search_results),
            )
            return [
                self._map_scored_point_to_result(point, include_vectors)
                for point in search_results
            ]
        except Exception as e:
            logger.error(
                "Search failed in collection '%s': %s", collection_name, e
            )
            raise

    def get_by_ids(self, collection_name: str, ids: List[str]) -> List[VectorRecord]:
        """
        Retrieve specific records by their IDs from Qdrant.
        """
        if self.client is None:
            logger.error("Attempted get_by_ids without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        logger.info(
            "Retrieving %d records by IDs from collection '%s'...",
            len(ids),
            collection_name,
        )
        qdrant_ids = [self._to_qdrant_id(rid) for rid in ids]

        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=qdrant_ids,
                with_payload=True,
                with_vectors=True,
            )
            logger.info("Found %d records.", len(points))
            return [self._map_point_to_record(point) for point in points]
        except Exception as e:
            logger.error(
                "Failed to retrieve records from collection '%s': %s",
                collection_name,
                e,
            )
            raise

    def count(self, collection_name: str) -> int:
        """
        Return the total number of records in the collection.
        """
        if self.client is None:
            logger.error("Attempted count without active connection.")
            raise RuntimeError("Database not connected. Call connect() first.")

        try:
            count_result = self.client.count(
                collection_name=collection_name, exact=True
            )
            logger.debug(
                "Count for collection '%s': %d", collection_name, count_result.count
            )
            return count_result.count
        except Exception as e:
            logger.error(
                "Failed to count records in collection '%s': %s",
                collection_name,
                e,
            )
            raise

    # Internal helper methods

    def _to_qdrant_id(self, record_id: str) -> str:
        """
        Convert any string to a Qdrant-compliant ID (UUID or integer).
        """
        if record_id.isdigit():
            return str(int(record_id))

        try:
            uuid.UUID(record_id)
            return record_id
        except ValueError:
            # Deterministically convert arbitrary string ID to a UUID
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))


    def _map_filters(self, filters: Dict[str, Any]) -> Any:
        """
        Map a flat key-value dictionary to Qdrant field conditions.
        """
        from qdrant_client.http import models as qmodels

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                qmodels.FieldCondition(
                    key=key, match=qmodels.MatchValue(value=value)
                )
            )
        return qmodels.Filter(must=must_conditions)

    def _map_point_to_record(self, point: Any) -> VectorRecord:
        """
        Map a Qdrant point to VectorRecord, restoring original ID if available.
        """
        payload = point.payload or {}
        original_id = payload.get("_original_id", str(point.id))
        clean_payload = {
            k: v for k, v in payload.items() if k != "_original_id"
        }

        vector = []
        if hasattr(point, "vector") and point.vector is not None:
            if isinstance(point.vector, list):
                vector = point.vector
            elif isinstance(point.vector, dict):
                vector = next(iter(point.vector.values()), [])

        return VectorRecord(
            id=original_id, vector=vector, payload=clean_payload
        )

    def _map_scored_point_to_result(
        self, point: Any, include_vector: bool
    ) -> SearchResult:
        """
        Map a Qdrant scored point to SearchResult, restoring original ID if available.
        """
        payload = point.payload or {}
        original_id = payload.get("_original_id", str(point.id))
        clean_payload = {
            k: v for k, v in payload.items() if k != "_original_id"
        }

        vector = None
        if include_vector and hasattr(point, "vector") and point.vector is not None:
            if isinstance(point.vector, list):
                vector = point.vector
            elif isinstance(point.vector, dict):
                vector = next(iter(point.vector.values()), None)

        return SearchResult(
            id=original_id,
            score=point.score,
            payload=clean_payload,
            vector=vector,
        )
