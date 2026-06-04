# rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.rag.Rag_module.rag_pipeline import RAGResult
from src.stores import GenerationConfig
from src.prompts import  normal_chat_system_prompt
from langchain_core.messages import HumanMessage


@dataclass
class NormalResult:
    query: str
    answer: str


class NormalPipeline:
    def __init__(
        self,
        generation_client,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.generation_client = generation_client
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.3, max_tokens=6000
        )
    def run(self ,query: str) -> NormalResult:

        # 1) Generate final answer
        answer = self._generate(query=query)
        print("\nRAG Response:\n", answer)

        return NormalResult(
            query=query,
            answer=answer,
        )

    def _generate(self, query: str) -> str:
       
        user_query = HumanMessage(content=query)
        
        full_history = [user_query]
        compiled_message = normal_chat_system_prompt.format_messages(
            chat_history= full_history,
        )
        response = self.generation_client.generate_text(
            messages=compiled_message,
            config=self.generation_config,
        )
        return response.content
