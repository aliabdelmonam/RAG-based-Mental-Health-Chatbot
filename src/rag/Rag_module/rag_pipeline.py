# rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional,List

from src.rag.Intent_Classifier_module.Intent_classifier import IntentClassifier
from src.stores import GenerationConfig
from src.db import Retrieve
from src.prompts import  rag_system_prompt
from langchain_core.messages import BaseMessage,HumanMessage
import tiktoken

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Takes a string of text and returns the number of tokens
    for a specific OpenAI model.
    """
    try:
        # Automatically gets the correct encoding for the specified model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a standard cl100k_base encoding if the model isn't recognized
        encoding = tiktoken.get_encoding("cl100k_base")
        
    # Encode the text and return the length of the token list
    return len(encoding.encode(text))

@dataclass
class RAGResult:
    query: str
    emotion: str
    retrieved_chunks: list
    context: str
    answer: str


class RAGPipeline:
    def __init__(
        self,
        generation_client,
        embedding_client,
        vector_db_client,
        collection_name: str,
        top_k: int = 5,
        intent_classifier: IntentClassifier = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vector_db_client = vector_db_client
        self.intent_classifier = intent_classifier
        self.collection_name = collection_name
        self.top_k = top_k
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.3, max_tokens=6000
        )
        self.retrieve = Retrieve(vector_db_client)

    def run(self,emotion:str ,query: str) -> RAGResult:
        
        # 2) Embed query
        embedding_query = self.embedding_client.embed_query(query)
        query_vector = embedding_query.embeddings[0]

        # 3) Retrieve top-k chunks
        search_results = self.retrieve.search_with(
            query_vector=query_vector,
            collection_name=self.collection_name,
            top_k=self.top_k,
            top_q=1
        )
        print("Search Results:")
        for r in search_results:
            payload_items = list(r.payload.items())[:5] if r.payload else []
            payload_str = ', '.join(f"{k}={v}" for k, v in payload_items)
            # print(f"- id={r.id} score={r.score:.4f} payload: {{{payload_str}}}")

        # 4) Build RAG context
        context = self._build_context(search_results)
        print(f"Total tokens in context and query: {count_tokens(context) + count_tokens(query)}")
        # 5) Generate final answer
        answer = self._generate(query=query, history=[], emotion=emotion, context=context)
        print("\nRAG Response:\n", answer)

        return RAGResult(
            query=query,
            emotion=emotion,
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

    def _generate(self, query: str, history: List[BaseMessage], emotion: str, context: str) -> str:
       
        user_query = HumanMessage(content=query)
        
        full_history = history + [user_query]
        compiled_message = rag_system_prompt.format_messages(
            context= context,
            chat_history= full_history,
            emotion= emotion
        )
        response = self.generation_client.generate_text(
            messages=compiled_message,
            config=self.generation_config,
        )
        return response.content

    @staticmethod
    def _safe_get(d: dict, key: str, default: str = "") -> str:
        v = d.get(key, default)
        return "" if v is None else str(v)