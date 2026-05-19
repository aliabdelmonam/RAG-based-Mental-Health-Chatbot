from enum import Enum

class LLMEnum(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GEOQ   =  "GROQ"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "CHATBOT"

class GroqEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
