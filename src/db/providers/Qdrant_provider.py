import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.core.logger import get_logger
from src.db.vector_db_interface import SearchResult, VectorDBInterface, VectorRecord

logger = get_logger("QDrantProvider")
QdrantClient.__del__ = lambda self: None

DISTANCE_MAP = {
    "cosine":      qm.Distance.COSINE,
    "euclidean":   qm.Distance.EUCLID,
    "dot_product": qm.Distance.DOT,
    "manhattan":   qm.Distance.MANHATTAN,
}


class QDrantProvider(VectorDBInterface):

    def __init__(
        self,
        embedding: Embeddings,           # ← real embedding client
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        in_memory: bool = False,
    ):
        self.embedding = embedding
        self.url       = url
        self.api_key   = api_key
        self.path      = path
        self.in_memory = in_memory

        self.client: Optional[QdrantClient] = None
        self._store:  Optional[QdrantVectorStore] = None

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
            self._store  = None
            logger.info("Qdrant disconnected.")

    # ------------------------------------------------------------------ #
    #  Collection management                                               #
    # ------------------------------------------------------------------ #

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self._require_connection()

        distance = DISTANCE_MAP.get(metric)
        if distance is None:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from: {list(DISTANCE_MAP)}")

        if not self._collection_exists(collection_name):
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
        self._store = None
        logger.info("Deleted collection '%s'.", collection_name)

    # ------------------------------------------------------------------ #
    #  CRUD                                                                #
    # ------------------------------------------------------------------ #

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        """Insert pre-computed vectors directly — embedding is NOT called here."""
        self._require_connection()

        points = [
            qm.PointStruct(
                id=self._to_qdrant_id(r.id),
                vector=r.vector,
                payload={
                    **(r.payload or {}),
                    "_original_id": r.id,
                    # LangChain reads page_content from payload when reconstructing Documents
                    "page_content": (r.payload or {}).get("page_content", ""),
                },
            )
            for r in records
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.debug("Upserted %d records into '%s'.", len(records), collection_name)

    def upsert_texts(self, collection_name: str, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        LangChain-style insert: give it raw text, it embeds and stores.
        This is the method that actually uses self.embedding.
        """
        self._require_connection()
        store = self._get_store(collection_name)
        store.add_texts(texts=texts, metadatas=metadatas or [{} for _ in texts])
        logger.debug("Upserted %d texts into '%s'.", len(texts), collection_name)

    def delete(self, collection_name: str, ids: List[str]) -> None:
        self._require_connection()
        qdrant_ids = [self._to_qdrant_id(i) for i in ids]
        self.client.delete(
            collection_name=collection_name,
            points_selector=qm.PointIdsList(points=qdrant_ids),
        )
        logger.debug("Deleted %d records from '%s'.", len(ids), collection_name)

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        collection_name: str,
        query_vector: List[float],          # pre-computed vector
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """Search with a pre-computed vector — embedding is NOT called."""
        self._require_connection()

        store = self._get_store(collection_name)
        lc_filter = self._build_filter(filters) if filters else None

        pairs: List[tuple[Document, float]] = store.similarity_search_by_vector_with_relevance_scores(
            embedding=query_vector,
            k=limit,
            filter=lc_filter,
        )

        return [self._doc_to_search_result(doc, score) for doc, score in pairs]

    def search_by_text(
        self,
        collection_name: str,
        query: str,                         # raw text — will be embedded
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search with raw text — embedding IS called here."""
        self._require_connection()

        store = self._get_store(collection_name)
        lc_filter = self._build_filter(filters) if filters else None

        pairs: List[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query=query,
            k=limit,
            filter=lc_filter,
        )

        return [self._doc_to_search_result(doc, score) for doc, score in pairs]

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
    #  LangChain extras                                                    #
    # ------------------------------------------------------------------ #

    def as_retriever(self, collection_name: str, k: int = 5):
        """Plug directly into any LCEL chain: retriever | prompt | llm"""
        return self._get_store(collection_name).as_retriever(search_kwargs={"k": k})

    def get_lc_store(self, collection_name: str) -> QdrantVectorStore:
        """Expose the raw LangChain store for advanced use."""
        return self._get_store(collection_name)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_store(self, collection_name: str) -> QdrantVectorStore:
        # One store instance is enough — it's just a wrapper, holds no data.
        if not self._store:
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding,
            )
        return self._store

    def _doc_to_search_result(self, doc: Document, score: float) -> SearchResult:
        payload = dict(doc.metadata)
        original_id = payload.pop("_original_id", payload.get("id", ""))
        payload.pop("page_content", None)
        return SearchResult(id=original_id, score=score, payload=payload, vector=None)

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
        import uuid
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
        v = getattr(point, "vector", None)
        vector = v if isinstance(v, list) else (next(iter(v.values()), []) if v else [])
        return VectorRecord(id=original_id, vector=vector, payload=payload)