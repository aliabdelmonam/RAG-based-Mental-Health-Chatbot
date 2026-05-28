# rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rag import IntentClassifier, LanguageDetector
from stores import GenerationConfig, Message


@dataclass
class RAGResult:
    query: str
    intent: str
    retrieved_chunks: list
    context: str
    answer: str


class RAGPipeline:
    def __init__(
        self,
        generation_client,
        embedding_client,
        vector_db_client,
        intent_classifier: IntentClassifier,
        collection_name: str,
        top_k: int = 5,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vector_db_client = vector_db_client
        self.intent_classifier = intent_classifier
        self.collection_name = collection_name
        self.top_k = top_k
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.3, max_tokens=50012
        )

    def run(self, query: str) -> RAGResult:
        # 1) Classify intent
        intent_raw = self.intent_classifier.classify(query)
        print(f"Intent Raw: {intent_raw}")

        # 2) Embed query
        embedding_query = self.embedding_client.embed_query(query)
        query_vector = embedding_query.embeddings[0]

        # 3) Retrieve top-k chunks
        search_results = self.vector_db_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k,
        )
        print("Search Results:")
        for r in search_results:
            payload_items = list(r.payload.items())[:5] if r.payload else []
            payload_str = ', '.join(f"{k}={v}" for k, v in payload_items)
            print(f"- id={r.id} score={r.score:.4f} payload: {{{payload_str}}}")

        # 4) Build RAG context
        context = self._build_context(search_results)

        # 5) Generate final answer
        answer = self._generate(query, intent_raw, context)
        print("\nRAG Response:\n", answer)

        return RAGResult(
            query=query,
            intent=intent_raw,
            retrieved_chunks=search_results,
            context=context,
            answer=answer,
        )

    def _build_context(self, search_results: list) -> str:
        context_blocks: list[str] = []
        for i, r in enumerate(search_results):
            payload = r.payload or {}
            question = self._safe_get(payload, "question")
            answer = self._safe_get(payload, "answer")
            block = (
                f"[CHUNK {i+1}]\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            ).strip()
            context_blocks.append(block)
        return "\n\n".join(context_blocks) if context_blocks else ""

    def _generate(self, query: str, intent_raw: str, context: str) -> str:
        system_prompt = (
            "You are a mental health assistant. "
            "Use the provided context to answer the user's question. "
            "Be empathetic, non-judgmental, and avoid diagnosing. "
            "If the user expresses crisis, encourage contacting local emergency services or a crisis hotline."
        )
        user_prompt = (
            f"User message: {query}\n\n"
            f"Intent (raw): {intent_raw}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer the user helpfully based on the context."
        )
        messages = [Message(role="user", content=user_prompt)]
        response = self.generation_client.generate_text(
            messages=messages,
            system_prompt=system_prompt,
            config=self.generation_config,
        )
        return response.content

    @staticmethod
    def _safe_get(d: dict, key: str, default: str = "") -> str:
        v = d.get(key, default)
        return "" if v is None else str(v)