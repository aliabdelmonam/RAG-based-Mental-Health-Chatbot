from .VectorDBENUM import VectorDBEnum
from .providers import ChromaProvider, QDrantProvider
from langchain_core.embeddings import Embeddings
from typing import Optional


class VectorDBFactory:
    
    def __init__(self,config):
        self.config = config

    def create(self, provider: VectorDBEnum, embedding: Optional[Embeddings] = None):
        if provider == VectorDBEnum.CHROMA.value:
            return ChromaProvider()
        elif provider == VectorDBEnum.QDRANT.value:
            return QDrantProvider(url=self.config.QDRANT_URL,api_key=self.config.QDRANT_API_KEY
                                  ,path=self.config.QDRANT_PATH,in_memory=self.config.QDRANT_IN_MEMORY,
                                  embedding=embedding)
        else:
            raise ValueError(f"Unknown provider: {provider}")