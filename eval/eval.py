# eval/eval.py
from __future__ import annotations
import os
import asyncio
import pandas as pd
from datasets import Dataset

# Ragas imports stay the same
from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
# LlamaIndex core Workflow components
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step

# Ensure project root is in sys.path for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your existing architectural imports
from src.core import get_settings
from src.stores import LLMProviderFactory, GenerationConfig
from src.db import VectorDBFactory
from src.rag import IntentClassifier, LanguageDetector, RAGPipeline

settings = get_settings()


from langchain_cohere import ChatCohere, CohereEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(ChatCohere(
    model="command-r-plus",   # or "command-r" for cheaper option
    cohere_api_key=settings.COHERE_API_KEY,
))

ragas_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=settings.COHERE_API_KEY,
))

# ─────────────────────────────────────────────────────────────
# 1. Define the Evaluation Workflow Class
# ─────────────────────────────────────────────────────────────
class RagasEvaluationWorkflow(Workflow):
    def __init__(
        self,
        pipeline: RAGPipeline,
        questions: list[str],
        ground_truths: list[str],
        batch_size: int = 5,        # how many questions per batch
        batch_delay: float = 2.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.questions = questions
        self.ground_truths = ground_truths
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    @step
    async def generate_rag_dataset(self, ev: StartEvent) -> StopEvent:
        """
        Executes evaluation rows in controlled batches to avoid
        rate-limiting, memory pressure, and API overload.

        Each batch runs concurrently internally; batches are separated
        by a configurable delay (batch_delay seconds).
        """

        async def process_single_question(question: str):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.pipeline.run, question)

            if result.context:
                blocks = [b.strip() for b in result.context.split("\n\n") if b.strip()]
            else:
                blocks = []

            return result.answer, blocks

        total = len(self.questions)
        all_answers: list[str] = []
        all_contexts: list[list[str]] = []

        # Split questions into chunks of batch_size
        batches = [
            self.questions[i : i + self.batch_size]
            for i in range(0, total, self.batch_size)
        ]
        total_batches = len(batches)

        print(f"🚀 Starting evaluation | {total} questions | "
              f"{total_batches} batches of {self.batch_size}")

        for batch_idx, batch in enumerate(batches, start=1):
            print(f"  ⏳ Batch {batch_idx}/{total_batches} "
                  f"({len(batch)} questions)...")

            tasks = [process_single_question(q) for q in batch]
            records = await asyncio.gather(*tasks)

            all_answers.extend(r[0] for r in records)
            all_contexts.extend(r[1] for r in records)

            print(f"  ✅ Batch {batch_idx}/{total_batches} done.")

            # Pause between batches (skip after the last one)
            if batch_idx < total_batches:
                print(f"  💤 Waiting {self.batch_delay}s before next batch...")
                await asyncio.sleep(self.batch_delay)

        print(f"🎯 All {total} questions processed.")

        ragas_dataset = Dataset.from_dict({
            "question":    self.questions,
            "answer":      all_answers,
            "contexts":    all_contexts,
            "ground_truth": self.ground_truths,
        })

        return StopEvent(result=ragas_dataset)


# ─────────────────────────────────────────────────────────────
# 2. Asynchronous Execution Wrapper
# ─────────────────────────────────────────────────────────────
async def run_evaluation():

    llm_provider = LLMProviderFactory(settings)
    vector_db_provider = VectorDBFactory(settings)

    generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
    embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)

    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)

    _ = embedding_client.health_check()
    _ = generation_client.health_check()

    qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND)
    qdrant_client.connect()

    eval_df = pd.read_csv(r"C:\Users\aliab\Desktop\RAG-based Mental Health Chatbot\eval\dataset\evaluation_test.csv").head(10)
    questions: list[str] = eval_df["synthetic_question"].astype(str).tolist()
    ground_truths: list[str] = eval_df["reference_answer"].astype(str).tolist()

    gen_config = GenerationConfig(max_tokens=6000, temperature=0.3)
    pipeline = RAGPipeline(
        generation_client=generation_client,
        embedding_client=embedding_client,
        vector_db_client=qdrant_client,
        collection_name="Normal_chunking",
        top_k=3,
        generation_config=gen_config,
    )

    eval_workflow = RagasEvaluationWorkflow(
        pipeline=pipeline,
        questions=questions,
        ground_truths=ground_truths,
        batch_size=5,       # ← tune this: lower = safer, higher = faster
        batch_delay=30.0,    # ← tune this: seconds to rest between batches
        timeout=300.0,
    )

    ragas_dataset = await eval_workflow.run()

    print("📊 Computing Ragas scores...")
    # ✅ correct indentation (1 level inside the function)
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
        ],
    )

    print("\n================ EVALUATION METRIC RESULTS ================")
    print(results)


if __name__ == "__main__":
    asyncio.run(run_evaluation())