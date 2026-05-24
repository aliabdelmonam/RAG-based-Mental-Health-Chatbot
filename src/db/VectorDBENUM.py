from enum import Enum
from qdrant_client.http import models as qmodels

class VectorDBEnum(Enum):
    CHROMA = "CHROMA"
    QDRANT = "QDRANT"

class QDrantVectorDB(Enum):
    PERSISTENT = "persistent"
    IN_MEMORY = "in_memory"
    COSINE = qmodels.Distance.COSINE
    EUCLIDEAN = qmodels.Distance.EUCLID
    DOT_PRODUCT = qmodels.Distance.DOT 
    MANHATTAN = qmodels.Distance.MANHATTAN
    
class CHROMAVectorDB(Enum):
    PERSISTENT = "persistent"
    IN_MEMORY = "in_memory"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotproduct"  
