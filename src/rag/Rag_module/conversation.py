# src/rag/history.py
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List
from src.core.logger import get_logger

logger = get_logger("Convserstation:")

@dataclass
class ConversationHistory:
    max_turns: int = 10  # keeps last N human+assistant pairs
    _messages: List[BaseMessage] = field(default_factory=list, repr=False)

    def add_user(self, text: str) -> None:
        self._messages.append(HumanMessage(content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append(AIMessage(content=text))
        self._trim()

    def get(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        # each turn = 1 human + 1 assistant = 2 messages
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
    
    def __str__(self) -> str:
        if not self._messages:
            return "ConversationHistory: empty"
        
        lines = [f"ConversationHistory ({len(self._messages)} messages):"]
        for i, msg in enumerate(self._messages):
            role = msg.__class__.__name__.replace("Message", "")
            content = msg.content if msg.content else "[tool call]"
            lines.append(f"  [{i+1}] {role}: {content[:100]}{'...' if len(str(content)) > 100 else ''}")
        
        return "\n".join(lines)
    
    def reduce_history(self, k: int = 2) -> List[BaseMessage]:
        """Returns the last k turns (k * 2 messages) without modifying the internal state."""
        messages_needed = k * 2
        total_messages = len(self._messages)
        
        if messages_needed >= total_messages:
            logger.warning(
                f"Requested {k} turns ({messages_needed} messages), but history only contains "
                f"{total_messages} messages. Returning full history."
            )
            return list(self._messages)
            
        # Return only the last k * 2 messages
        logger.info(f"Cutting the history to {k*2}")
        return list(self._messages[-messages_needed:])