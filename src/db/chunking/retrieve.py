
from src.db.vector_db_interface import SearchResult
from typing import  List
# import tiktoken

class Retrieve:

    def __init__(self,vector_db_client):
        self.vector_db_client = vector_db_client
    

    # def _get_first_n_tokens_as_text(text: str, n: int, model_name: str = "gpt-4") -> str:
    #     try:
    #         encoding = tiktoken.encoding_for_model(model_name)
    #     except KeyError:
    #         encoding = tiktoken.get_encoding("cl100k_base")
            
    #     # Encode to tokens, slice the first n, then decode back to string
    #     tokens = encoding.encode(text)
    #     first_n_tokens = tokens[:n]
    #     return encoding.decode(first_n_tokens)

    def search_with(self, query_vector, collection_name="Normal_chunking", top_k=5, 
                 top_q=2,chunk_size_token:int=500) -> List[SearchResult]:
        search_results = self.vector_db_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        cnt = 0
        for i,r in enumerate(search_results):
            answers = r.payload.get('answer',[])
            if len(answers) >= (top_q - cnt):
                # meaning that we will not take all answers
                search_results[i].payload['answer'] = answers[:top_q - cnt]
                return search_results[:i+1]
            else:
                cnt += len(answers)
        return search_results