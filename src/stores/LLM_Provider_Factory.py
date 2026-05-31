from .LLMEnums import LLMEnum
from .providers import CohereLLMProvider, GroqLLMProvider, HuggingFaceLLMProvider,OpenAILLMProvider,ColabLLMProvider


class LLMProviderFactory:
    
    def __init__(self,config):
        self.config = config

    def create(self,provider:LLMEnum):
        if provider == LLMEnum.OPENAI.value:
            return OpenAILLMProvider()
        elif provider == LLMEnum.COHERE.value:
            return CohereLLMProvider()
        elif provider == LLMEnum.GROQ.value:
            return GroqLLMProvider(api_key=self.config.GROQ_API_KEY, generation_model=self.config.GENERATION_MODEL_ID)
        elif provider == LLMEnum.HUGGINGFACE.value:
            return HuggingFaceLLMProvider(api_key=self.config.HF_API_KEY)
        elif provider == LLMEnum.COLAB.value:                # ← add this
            return ColabLLMProvider(
                base_url=self.config.COLAB_NGROK_URL,
                generation_model=self.config.GENERATION_MODEL_ID,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")