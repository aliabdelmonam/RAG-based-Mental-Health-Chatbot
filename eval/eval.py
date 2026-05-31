# eval/eval.py
from __future__ import annotations
import os
import asyncio
import pandas as pd
from datasets import Dataset

# Ragas imports stay the same
from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
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
# Example: Adapting max_tokens for GenerationConfig
# config = GenerationConfig(max_tokens=2048)
# Pass 'config' to your pipeline or generation call as needed

# ─────────────────────────────────────────────────────────────
# 1. Define the Evaluation Workflow Class
# ─────────────────────────────────────────────────────────────
class RagasEvaluationWorkflow(Workflow):
    def __init__(self, pipeline: RAGPipeline, questions: list[str], ground_truths: list[str], **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.questions = questions
        self.ground_truths = ground_truths

    @step
    async def generate_rag_dataset(self, ev: StartEvent) -> StopEvent:
        """
        Executes all evaluation rows concurrently using asyncio.gather.
        This provides a significant speedup over sequential python loops.
        """
        
        # Define a helper function to run a single row matching your logic
        async def process_single_question(question: str):
            # If your custom pipeline.run() is synchronous, run it in a thread executor 
            # to prevent it from blocking the main event loop.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.pipeline.run, question)
            
            # Cleanly slice contexts exactly like your previous dataframe setup
            if result.context:
                blocks = [b.strip() for b in result.context.split("\n\n") if b.strip()]
            else:
                blocks = []
                
            return result.answer, blocks

        print(f"🚀 Launching evaluation workflow for {len(self.questions)} questions concurrently...")
        
        # Package and execute tasks concurrently
        tasks = [process_single_question(q) for q in self.questions]
        records = await asyncio.gather(*tasks)
        
        # Unzip your parallel execution arrays
        answers = [r[0] for r in records]
        contexts = [r[1] for r in records]
        
        # Build the structured Ragas payload
        ragas_dataset = Dataset.from_dict({
            "question": self.questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": self.ground_truths,
        })
        
        return StopEvent(result=ragas_dataset)

# ─────────────────────────────────────────────────────────────
# 2. Asynchronous Execution Wrapper
# ─────────────────────────────────────────────────────────────
async def run_evaluation():
    settings = get_settings()

    # Maintain your custom Factories and Providers contract
    llm_provider = LLMProviderFactory(settings)
    vector_db_provider = VectorDBFactory(settings)

    generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
    embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)

    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)

    # Underlying connectivity validations
    _ = embedding_client.health_check()
    _ = generation_client.health_check()

    qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND)
    qdrant_client.connect()

    # Load file variables
    eval_df = pd.read_csv(r"C:\Users\BS\Desktop\RAG-based Mental Health Chatbot\eval\dataset\evaluation_test.csv")
    questions: list[str] = eval_df["synthetic_question"].astype(str).tolist()
    ground_truths: list[str] = eval_df["reference_answer"].astype(str).tolist()

    # Build modules natively exactly as defined
    lang_detector = LanguageDetector(
        model_path=r"C:\Users\BS\Downloads\language_detector.pkl",
        threshold=0.60,
    )
    intent_cls = IntentClassifier(
        generation_client=generation_client,
        language_detector=lang_detector,
    )

    # Set a custom max_tokens for generation
    gen_config = GenerationConfig(max_tokens=20048, temperature=0.3)
    pipeline = RAGPipeline(
        generation_client=generation_client,
        embedding_client=embedding_client,
        vector_db_client=qdrant_client,
        # intent_classifier=intent_cls,
        collection_name="Normal_chunking",
        top_k=2,
        generation_config=gen_config,
    )

    # Instantiate the LlamaIndex Workflow instance
    eval_workflow = RagasEvaluationWorkflow(
        pipeline=pipeline, 
        questions=questions, 
        ground_truths=ground_truths,
        timeout=150.0 # Adjust workflow execution window bounds based on dataset size
    )

    # Kick off Workflow execution context
    ragas_dataset = await eval_workflow.run()

    # Pass the compiled dataset straight to Ragas evaluator
    print("📊 Computing Ragas scores...")
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n================ EVALUATION METRIC RESULTS ================")
    print(results)


if __name__ == "__main__":
    # Workflows require an async entry point execution loop
    asyncio.run(run_evaluation())