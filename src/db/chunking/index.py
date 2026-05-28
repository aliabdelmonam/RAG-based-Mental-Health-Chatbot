import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from chunking import ChunkingStrategy
from stores import LLMProviderFactory
from core import get_settings
from db import  VectorDBFactory

settings = get_settings()


from db.vector_db_interface import VectorRecord


# Initialize embedding provider
llm_provider = LLMProviderFactory(settings)
embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)
embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)


vector_db_provider = VectorDBFactory(settings)
qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND)
qdrant_client.connect() 
# if not embedding_client.health_check():
    # raise RuntimeError("Embedding client health_check() failed.")

embedding_dim = embedding_client.get_embedding_dimension()

# Build chunks from the source CSV
chunking_strategy = ChunkingStrategy(path_to_csv=r"C:\Users\BS\Desktop\mental_health_counseling_conversations.csv")
chunks = chunking_strategy.chunk_text()

if not chunks:
    raise RuntimeError("No chunks were generated from the CSV.")

# Ensure Qdrant collection exists
qdrant_client.create_collection(
    collection_name="Normal_chunking",
    dimension=embedding_dim,
    metric="cosine",
)

# Embed all chunk contents
chunk_texts = [c.content for c in chunks]
embedding_resp = embedding_client.embed_documents(chunk_texts)

vectors = embedding_resp.embeddings
if len(vectors) != len(chunks):
    raise RuntimeError(
        f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks."
    )

# Upsert records with the same payload/meta design already used in the commented example
source_name = "mental_health_counseling_conversations.csv"
records: list[VectorRecord] = []

for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
    records.append(
        VectorRecord(
            id=f"chunk_{i}",
            vector=vector,
            payload={
                "question": chunk.content,
                "answer": chunk.metadata.get("all_answers"),
                "chunk_index": i,
                "source": source_name,
                # also keep existing meta fields
                **chunk.metadata,
            },
        )
    )


qdrant_client.upsert(collection_name="Normal_chunking", records=records)

qdrant_client.disconnect()