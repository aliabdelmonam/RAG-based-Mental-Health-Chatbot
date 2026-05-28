from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class TextChunk:
    content: str  # The Question text (The only thing being embedded)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.content,
            "metadata": self.metadata
        }

class ChunkingStrategy:
    def __init__(self, path_to_csv: str):
        self.data = pd.read_csv(path_to_csv)

    def chunk_text(self, text: str = "") -> List[TextChunk]:
        questions = self.data['Question'].to_numpy()
        answers = self.data['Answers'].to_numpy()

        chunks = []

        for idx, (q, a) in enumerate(zip(questions, answers)):
            # Ensure 'a' is a clean Python list of strings (not a numpy array or bracketed string)
            if isinstance(a, (np.ndarray, tuple)):
                answer_list = a.tolist()
            elif isinstance(a, list):
                answer_list = a
            else:
                answer_list = [str(a)]

            # Force all list elements to clean strings to prevent Qdrant errors
            clean_answer_list = [str(item) for item in answer_list]

            # Store the whole list under a single point
            chunks.append(TextChunk(
                content=str(q), 
                metadata={
                    "all_answers": clean_answer_list,
                    "total_answers_count": len(clean_answer_list),
                    "parent_question_id": idx
                }
            ))
        return chunks