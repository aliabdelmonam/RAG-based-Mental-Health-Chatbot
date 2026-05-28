import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from core.logger import get_logger
from db.vector_db_interface import SearchResult, VectorDBInterface, VectorRecord

logger = get_logger("QDrantProvider")
QdrantClient.__del__ = lambda self: None
# Metric name → Qdrant Distance enum
DISTANCE_MAP = {
    "cosine":      qm.Distance.COSINE,
    "euclidean":   qm.Distance.EUCLID,
    "dot_product": qm.Distance.DOT,
    "manhattan":   qm.Distance.MANHATTAN,
}


class QDrantProvider(VectorDBInterface):
    """
    Qdrant implementation of VectorDBInterface.

    Modes (chosen automatically from arguments / env vars):
      • in-memory  – fast, no persistence; good for testing
      • local      – persists to disk at `path`
      • cloud      – connects to a remote Qdrant instance via `url` + optional `api_key`
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        in_memory: bool = False,
    ):
        # Resolve from env vars if not provided directly
        self.url     = url     
        self.api_key = api_key
        self.path    = path   
        self.in_memory = in_memory

        self.client: Optional[QdrantClient] = None

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self.client:
            return

        if self.in_memory:
            logger.info("Connecting to Qdrant: in-memory")
            self.client = QdrantClient(location=":memory:")

        elif self.url:
            logger.info("Connecting to Qdrant: cloud (%s)", self.url)
            self.client = QdrantClient(url=self.url, api_key=self.api_key)

        else:
            persist_path = self.path or os.path.join(os.getcwd(), "data", "qdrant_db")
            logger.info("Connecting to Qdrant: local (%s)", persist_path)
            self.client = QdrantClient(path=persist_path)

    def disconnect(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning("Error while closing Qdrant client: %s", e)

            self.client = None
            logger.info("Qdrant disconnected.")

    # ------------------------------------------------------------------ #
    #  Collection management                                               #
    # ------------------------------------------------------------------ #

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self._require_connection()

        distance = DISTANCE_MAP.get(metric)
        if distance is None:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from: {list(DISTANCE_MAP)}")

        exists = self._collection_exists(collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=dimension, distance=distance),
            )
            logger.info("Created collection '%s' (dim=%d, metric=%s).", collection_name, dimension, metric)
        else:
            logger.info("Collection '%s' already exists.", collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self._require_connection()
        self.client.delete_collection(collection_name=collection_name)
        logger.info("Deleted collection '%s'.", collection_name)

    # ------------------------------------------------------------------ #
    #  CRUD                                                                #
    # ------------------------------------------------------------------ #

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        self._require_connection()

        points = [
            qm.PointStruct(
                id=self._to_qdrant_id(r.id),
                vector=r.vector,
                payload={**(r.payload or {}), "_original_id": r.id},
            )
            for r in records
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.debug("Upserted %d records into '%s'.", len(records), collection_name)

    def delete(self, collection_name: str, ids: List[str]) -> None:
        self._require_connection()
        qdrant_ids = [self._to_qdrant_id(i) for i in ids]
        self.client.delete(
            collection_name=collection_name,
            points_selector=qm.PointIdsList(points=qdrant_ids),
        )
        logger.debug("Deleted %d records from '%s'.", len(ids), collection_name)

    def search(
    self,
    collection_name: str,
    query_vector: List[float],
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_vectors: bool = False,
) -> List[SearchResult]:
        self._require_connection()

        query_filter = self._build_filter(filters) if filters else None

        if hasattr(self.client, "query_points"):
            # Newer qdrant-client API (v1.7+)
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=include_vectors,
                query_filter=query_filter,
            )
            results = response.points  # ← QueryResponse wraps the list here

        elif hasattr(self.client, "search"):
            # Older qdrant-client API (pre-v1.7)
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=include_vectors,
            )

        else:
            raise AttributeError(
                "Unsupported qdrant-client version: neither 'query_points' nor 'search' found."
            )

        return [self._to_search_result(p, include_vectors) for p in results]

    def get_by_ids(self, collection_name: str, ids: List[str]) -> List[VectorRecord]:
        self._require_connection()
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[self._to_qdrant_id(i) for i in ids],
            with_payload=True,
            with_vectors=True,
        )
        return [self._to_vector_record(p) for p in points]

    def count(self, collection_name: str) -> int:
        self._require_connection()
        return self.client.count(collection_name=collection_name, exact=True).count

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _require_connection(self) -> None:
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

    def _collection_exists(self, name: str) -> bool:
        try:
            if hasattr(self.client, "collection_exists"):
                return self.client.collection_exists(name)
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def _to_qdrant_id(self, record_id: str) -> str:
        """Convert any string ID to a UUID (Qdrant-compatible)."""
        if record_id.isdigit():
            return str(int(record_id))
        try:
            uuid.UUID(record_id)
            return record_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))

    def _build_filter(self, filters: Dict[str, Any]) -> qm.Filter:
        return qm.Filter(
            must=[
                qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                for k, v in filters.items()
            ]
        )

    def _to_vector_record(self, point: Any) -> VectorRecord:
        payload = point.payload or {}
        original_id = payload.pop("_original_id", str(point.id))
        vector = self._extract_vector(point)
        return VectorRecord(id=original_id, vector=vector, payload=payload)

    def _to_search_result(self, point: Any, include_vector: bool) -> SearchResult:
        payload = dict(point.payload or {})
        original_id = payload.pop("_original_id", str(point.id))
        vector = self._extract_vector(point) if include_vector else None
        return SearchResult(id=original_id, score=point.score, payload=payload, vector=vector)

    @staticmethod
    def _extract_vector(point: Any) -> list:
        v = getattr(point, "vector", None)
        if v is None:
            return []
        return v if isinstance(v, list) else next(iter(v.values()), [])