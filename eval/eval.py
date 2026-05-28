# eval/eval.py
from __future__ import annotations

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.core import get_settings
from src.stores import LLMProviderFactory
from src.db import VectorDBFactory
from src.rag import IntentClassifier, LanguageDetector, RAGPipeline


def main() -> None:
    settings = get_settings()

    # 1) Providers
    llm_provider = LLMProviderFactory(settings)
    vector_db_provider = VectorDBFactory(settings)

    generation_client = llm_provider.create(provider=settings.GENERATION_BACKEND)
    embedding_client = llm_provider.create(provider=settings.EMBEDDING_BACKEND)

    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID)

    # Health checks
    _ = embedding_client.health_check()
    _ = generation_client.health_check()

    # Connect to vector DB
    qdrant_client = vector_db_provider.create(provider=settings.VECTORDB_BACKEND)
    qdrant_client.connect()

    # 2) Dataset
    eval_df = pd.read_csv(r"eval\evaluation_test.csv")
    questions: list[str] = eval_df["synthetic_question"].astype(str).tolist()
    ground_truths: list[str] = eval_df["reference_answer"].astype(str).tolist()

    # 3) Build RAG pipeline (single-call)
    lang_detector = LanguageDetector(
        model_path=r"C:\Users\BS\Downloads\language_detector.pkl",
        threshold=0.60,
    )
    intent_cls = IntentClassifier(
        generation_client=generation_client,
        language_detector=lang_detector,
    )

    pipeline = RAGPipeline(
        generation_client=generation_client,
        embedding_client=embedding_client,
        vector_db_client=qdrant_client,
        intent_classifier=intent_cls,
        collection_name="Normal_chunking",
        top_k=5,
    )

    # 4) Run evaluation generation step
    answers: list[str] = []
    contexts: list[list[str]] = []

    for question in questions:
        result = pipeline.run(question)
        answers.append(result.answer)

        # ragas expects contexts to be: List[List[str]] per sample
        # We split the pipeline context into chunk blocks.
        if result.context:
            blocks = [b.strip() for b in result.context.split("\n\n") if b.strip()]
        else:
            blocks = []
        contexts.append(blocks)

    # 5) Compile ragas dataset
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ragas metric computation
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n================ EVALUATION METRIC RESULTS ================")
    print(results)


if __name__ == "__main__":
    main()

