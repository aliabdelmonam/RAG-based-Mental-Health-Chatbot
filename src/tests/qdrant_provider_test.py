import pytest
from src.db.providers.Qdrant_provider import QDrantProvider
from src.db.VectorDBENUM import QDrantVectorDB
from src.db.vector_db_interface import VectorRecord

# Skip all tests if qdrant-client is not installed
try:
    import qdrant_client
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client is not installed")


@pytest.fixture
def qdrant_provider():
    # Use the QDrantVectorDB enum for in-memory setup
    provider = QDrantProvider(location=QDrantVectorDB.IN_MEMORY)
    provider.connect()
    yield provider
    provider.disconnect()


def test_qdrant_connection():
    provider = QDrantProvider(location=QDrantVectorDB.IN_MEMORY)
    assert provider.client is None
    provider.connect()
    assert provider.client is not None
    provider.disconnect()
    assert provider.client is None


def test_create_and_delete_collection(qdrant_provider):
    coll_name = "test_collection"
    # Use the QDrantVectorDB enum for the metric definition
    qdrant_provider.create_collection(
        coll_name, dimension=3, metric=QDrantVectorDB.COSINE
    )

    # Verify collection exists using the client
    exists = qdrant_provider.client.collection_exists(coll_name)
    assert exists

    # Delete collection
    qdrant_provider.delete_collection(coll_name)
    assert not qdrant_provider.client.collection_exists(coll_name)


def test_upsert_count_and_get(qdrant_provider):
    coll_name = "test_collection"
    qdrant_provider.create_collection(
        coll_name, dimension=3, metric=QDrantVectorDB.COSINE
    )

    records = [
        VectorRecord(id="doc_1", vector=[1.0, 0.0, 0.0], payload={"tag": "greet"}),
        VectorRecord(id="doc_2", vector=[0.0, 1.0, 0.0], payload={"tag": "help"}),
    ]

    qdrant_provider.upsert(coll_name, records)
    assert qdrant_provider.count(coll_name) == 2

    # Get by original string IDs
    retrieved = qdrant_provider.get_by_ids(coll_name, ["doc_1", "doc_2"])
    assert len(retrieved) == 2

    # Verify original IDs and payload are restored correctly
    ids = {r.id for r in retrieved}
    assert ids == {"doc_1", "doc_2"}

    doc1 = next(r for r in retrieved if r.id == "doc_1")
    assert doc1.payload == {"tag": "greet"}
    assert doc1.vector == [1.0, 0.0, 0.0]


def test_search_similarity(qdrant_provider):
    coll_name = "test_collection"
    qdrant_provider.create_collection(
        coll_name, dimension=3, metric=QDrantVectorDB.COSINE
    )

    records = [
        VectorRecord(id="doc_1", vector=[1.0, 0.0, 0.0], payload={"type": "A"}),
        VectorRecord(id="doc_2", vector=[0.9, 0.1, 0.0], payload={"type": "B"}),
        VectorRecord(id="doc_3", vector=[0.0, 1.0, 0.0], payload={"type": "A"}),
    ]
    qdrant_provider.upsert(coll_name, records)

    # Search with limit 2 close to doc_1 [1.0, 0.0, 0.0]
    results = qdrant_provider.search(
        coll_name, query_vector=[1.0, 0.0, 0.0], limit=2
    )
    assert len(results) == 2
    assert results[0].id == "doc_1"
    assert results[1].id == "doc_2"
    assert results[0].vector is None

    # Search with include_vectors=True
    results_with_vectors = qdrant_provider.search(
        coll_name, query_vector=[1.0, 0.0, 0.0], limit=1, include_vectors=True
    )
    assert results_with_vectors[0].vector == [1.0, 0.0, 0.0]

    # Search with metadata filtering
    filtered_results = qdrant_provider.search(
        coll_name, query_vector=[1.0, 0.0, 0.0], limit=3, filters={"type": "A"}
    )
    assert len(filtered_results) == 2
    assert {r.id for r in filtered_results} == {"doc_1", "doc_3"}


def test_delete_vectors(qdrant_provider):
    coll_name = "test_collection"
    qdrant_provider.create_collection(
        coll_name, dimension=2, metric=QDrantVectorDB.COSINE
    )

    records = [
        VectorRecord(id="doc_1", vector=[1.0, 0.0]),
        VectorRecord(id="doc_2", vector=[0.0, 1.0]),
    ]
    qdrant_provider.upsert(coll_name, records)
    assert qdrant_provider.count(coll_name) == 2

    # Delete one
    qdrant_provider.delete(coll_name, ["doc_1"])
    assert qdrant_provider.count(coll_name) == 1

    retrieved = qdrant_provider.get_by_ids(coll_name, ["doc_1", "doc_2"])
    assert len(retrieved) == 1
    assert retrieved[0].id == "doc_2"
