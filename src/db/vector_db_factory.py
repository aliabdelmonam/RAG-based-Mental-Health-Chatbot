from .VectorDBENUM import VectorDBEnum
from .providers import ChromaProvider, QDrantProvider


class VectorDBFactory:
    
    def create(self,provider:VectorDBEnum):
        if provider == VectorDBEnum.CHROMA:
            return ChromaProvider()
        elif provider == VectorDBEnum.QDRANT:
            return QDrantProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")