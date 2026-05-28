# eval/evaluate_chunks_ragas.py
import os
import sys
import torch
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import pandas as pd


# =====================================================================
# STEP 1: Import dataset
# =====================================================================
try:
    eval_df = pd.read_csv(r'eval\evaluation_test.csv')
except Exception as e:
    print(f"Error loading evaluation dataset: {e}")
questions = eval_df["synthetic_question"].tolist()
ground_truths = eval_df["reference_answer"].tolist()

# =====================================================================
# STEP 4: Run the RAG Generation Step Across Collections
# =====================================================================
# Let's say you want to evaluate your "fixed_256" chunking strategy collection
collection_to_test = "docs_fixed_256"

answers = []
contexts = []

print(f"\nGenerating system answers using collection: '{collection_to_test}'...")

for question in questions:
    # A. Retrieve context segments using your Qdrant provider search method
    search_results = db.search(
        collection_name=collection_to_test,
        query_vector=local_embeddings.embed_query(question),
        limit=2
    )
    
    # Extract text content strings from payloads
    retrieved_chunks = [res.payload.get("text", "") for res in search_results]
    
    # B. Generate a response using your local LLM pipeline (Simulating your RAG generation)
    context_str = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context_str}\n\nQuestion: {row['question']}\nAnswer:"
    generated_output = hf_pipeline.invoke(prompt)
    
    # Accumulate evaluation columns
    questions.append(row["question"])
    ground_truths.append(row["ground_truth"])
    contexts.append(retrieved_chunks)
    answers.append(generated_output)

# Close database file locks cleanly
db.disconnect()

# =====================================================================
# STEP 5: Compile and Execute Ragas Evaluation Benchmark
# =====================================================================
# Format into a Hugging Face Dataset Object
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
ragas_dataset = Dataset.from_dict(data_dict)

print("\nExecuting Ragas metric matrix calculation...")
results = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n================ EVALUATION METRIC RESULTS ================")
print(results)