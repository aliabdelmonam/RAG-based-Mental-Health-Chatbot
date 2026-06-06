from .VectorDBENUM import VectorDBEnum
from .providers import ChromaProvider, QDrantProvider


class VectorDBFactory:
    
    def __init__(self,config):
        self.config = config

    def create(self,provider:VectorDBEnum):
        if provider == VectorDBEnum.CHROMA.value:
            return ChromaProvider()
        elif provider == VectorDBEnum.QDRANT.value:
            return QDrantProvider(url=self.config.QDRANT_URL,api_key=self.config.QDRANT_API_KEY
                                  ,path=self.config.QDRANT_PATH,in_memory=self.config.QDRANT_IN_MEMORY,
                                  embedding=None)
        else:
            raise ValueError(f"Unknown provider: {provider}")