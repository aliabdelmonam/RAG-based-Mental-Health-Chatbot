from .LLMEnums import LLMEnum
from .providers import OpenAILLMProvider, CohereLLMProvider, GroqLLMProvider, HuggingFaceLLMProvider


class LLMProviderFactory:
    
    def create(self,provider:LLMEnum):
        if provider == LLMEnum.OPENAI:
            return OpenAILLMProvider()
        elif provider == LLMEnum.COHERE:
            return CohereLLMProvider()
        elif provider == LLMEnum.GROQ:
            return GroqLLMProvider()
        elif provider == LLMEnum.HUGGINGFACE:
            return HuggingFaceLLMProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")