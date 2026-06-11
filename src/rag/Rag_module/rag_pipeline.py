# rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional,List, TYPE_CHECKING
from src.stores import GenerationConfig
from src.prompts import  rag_system_prompt
from langchain_core.messages import BaseMessage,HumanMessage
import tiktoken
from src.stores.providers import crisis_tool
from langchain_core.messages import AIMessage, ToolMessage
from src.core.logger import get_logger
from src.rag.Rag_module.conversation import ConversationHistory
from src.rag.Rag_module.base_pipeline import BundleManager

if TYPE_CHECKING:
    from src.rag.Intent_Classifier_module.Intent_classifier import IntentClassifier

logger = get_logger(f"RagPipeline:")

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Takes a string of text and returns the number of tokens
    for a specific OpenAI model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    return len(encoding.encode(text))

@dataclass
class RAGResult:
    query: str
    emotion: str
    context: str
    answer: str


class RAGPipeline:
    def __init__(
        self,
        collection_name: str,
        top_k: int = 5,
        client = BundleManager,
        intent_classifier: IntentClassifier = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.intent_classifier = intent_classifier
        self.collection_name = collection_name
        self.top_k = top_k
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.3, max_tokens=6000
        )
        self.client = client

    def run(self, emotion: str, query: str, history: ConversationHistory) -> RAGResult:

        # 2) Embed query
        embedding_query = self.client.embed(query)
        query_vector = embedding_query

        # 3) Retrieve top-k chunks
        search_results = self.client.search(
            query_vector=query_vector,
            collection_name=self.collection_name,
            top_k=self.top_k,
            top_q=1
        )
        print("Search Results:")
        for r in search_results:
            payload_items = list(r.payload.items())[:5] if r.payload else []
            payload_str = ', '.join(f"{k}={v}" for k, v in payload_items)

        # 4) Build RAG context
        context = self._build_context(search_results)
        print(f"Total tokens in context and query: {count_tokens(context) + count_tokens(query)}")
        print("Search Results:\n", search_results)
        print("Search Results Context:\n", context)
        lc_history = history.get()

        # 5) First LLM call — LLM may respond normally or request a tool call
        answer = self._generate(query=query, history=lc_history, emotion=emotion, context=context, enable_tools=True)

        if answer.finish_reason == "tool_call":
            tool_calls = answer.content  # list of tool call dicts from the LLM

            # Step 1: append LLM's tool_call request to history
            lc_history.append(AIMessage(content="", tool_calls=tool_calls))

            # Step 2: execute each tool and append results to history
            for tool_call in tool_calls:
                logger.info(f"Executing tool: {tool_call['name']} | args: {tool_call['args']}")
                result = crisis_tool.invoke(tool_call["args"])  # actually runs the Python function
                logger.info(f"Tool result: {result}")
                lc_history.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))

            # Step 3: second LLM call — LLM now sees the tool result and responds to the user
            answer = self._generate(query=query, history=lc_history, emotion=emotion, context=context, enable_tools=False)

        history.add_user(query)
        history.add_assistant(answer.content)
        print("\nRAG Response:\n", answer.content)
        print(history)
        return RAGResult(
            query=query,
            emotion=emotion,
            context=context,
            answer=answer.content,
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

    def _generate(self, query: str, history: List[BaseMessage], emotion: str, context: str,enable_tools: bool = False) -> str:
        """
        Builds the full message list and calls the LLM.
        Returns the full GenerationResponse (not just content)
        so the caller can inspect finish_reason and tool_calls.
        """
        user_query = HumanMessage(content=query)
        full_history = history + [user_query]
        compiled_message = rag_system_prompt.format_messages(
            context=context,
            chat_history=full_history,
            emotion=emotion
        )
        response = self.client.generate(
            messages=compiled_message,
            config=self.generation_config,
            enable_tools=enable_tools
        )
        return response  # full GenerationResponse, not response.content

    @staticmethod
    def _safe_get(d: dict, key: str, default: str = "") -> str:
        v = d.get(key, default)
        return "" if v is None else str(v)