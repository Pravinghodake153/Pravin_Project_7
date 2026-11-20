"""
============================================================
main.py — Project 7 RAG System Entry Point
============================================================
PURPOSE:
    This is the file you run to test the RAG system.
    It runs all 5 assessment questions and prints the results.

HOW TO RUN:
    python3 main.py

WHAT IT DOES:
    1. Initializes the RAG engine (loads PDF, creates embeddings)
    2. Runs each of the 5 assessment questions
    3. Prints each answer with source page numbers
    4. Shows a summary table at the end
============================================================
"""

import logging
import sys
import json

# ----------------------------------------------------------
# Set up logging
# ----------------------------------------------------------
# logging.INFO shows important messages like progress updates.
# Change to logging.DEBUG to see every detail (useful for debugging).
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """
    Runs all 5 assessment questions and displays results.
    """
    print("=" * 70)
    print("  PROJECT 7: AI Research Paper & Architecture Literature Assistant")
    print("  RAG System — Retrieval Augmented Generation")
    print("=" * 70)
    print()

    # ----------------------------------------------------------
    # Step 1: Import and initialize the RAG engine
    # ----------------------------------------------------------
    # We import here (not at the top) so the logging setup
    # is already configured when the engine initializes.
    # ----------------------------------------------------------
    print("[Step 1] Initializing RAG Engine...")
    print("         (First run: downloads model + processes PDF)")
    print("         (Subsequent runs: loads from disk — much faster!)")
    print()

    from rag_engine import answer

    print()
    print("[Step 2] RAG Engine initialized successfully!")
    print()

    # ----------------------------------------------------------
    # Step 2: Define all 5 assessment questions
    # ----------------------------------------------------------
    questions = [
        {
            "id": "Q1",
            "question": (
                "What exact learning rate, number of warmup steps, "
                "and Adam optimizer beta values were used in the "
                "primary pre-training run described in Section 4?"
            ),
            "expected": (
                "Learning rate: 3e-4 (0.0003), Warmup steps: 4,000, "
                "Adam beta_1 = 0.9, beta_2 = 0.95 (from Page 16)"
            )
        },
        {
            "id": "Q2",
            "question": (
                "Who is the current Chief Executive Officer of OpenAI?"
            ),
            "expected": "[not found] (not in the PDF)"
        },
        {
            "id": "Q3",
            "question": (
                "By what percentage did the proposed hybrid retrieval "
                "architecture improve Exact Match (EM) accuracy on the "
                "benchmark dataset compared with the dense-only baseline?"
            ),
            "expected": (
                "14.6% improvement (from 61.2% to 75.8%) (from Page 31)"
            )
        },
        {
            "id": "Q4",
            "question": (
                "What GPU cluster configuration (number and model of "
                "GPUs) and total training duration, in days, were "
                "required to train the 13-billion-parameter model?"
            ),
            "expected": (
                "128 NVIDIA H100 (80GB SXM5) GPUs, 18.5 days (from Page 45)"
            )
        },
        {
            "id": "Q5",
            "question": (
                "What was the worldwide box-office revenue of the "
                "film Oppenheimer in 2023?"
            ),
            "expected": "[not found] (not in the PDF)"
        }
    ]

    # ----------------------------------------------------------
    # Step 3: Run each question through the RAG system
    # ----------------------------------------------------------
    results = []

    for q_data in questions:
        q_id = q_data["id"]
        question = q_data["question"]
        expected = q_data["expected"]

        print("-" * 70)
        print(f"  {q_id}")
        print(f"  Question: {question}")
        print()

        # Call the RAG system's answer function
        result = answer(question)

        print(f"  Answer:   {result['answer']}")
        print(f"  Sources:  {result['sources']}")
        print(f"  Expected: {expected}")
        print()

        results.append({
            "id": q_id,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "expected": expected
        })

    # ----------------------------------------------------------
    # Step 4: Print summary table
    # ----------------------------------------------------------
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'ID':<4} {'Sources':<20} {'Status':<12} {'Answer Preview'}")
    print(f"  {'--':<4} {'-------':<20} {'------':<12} {'-' * 30}")

    for r in results:
        answer_preview = r["answer"][:45] + "..." if len(r["answer"]) > 45 else r["answer"]
        is_found = r["answer"] != "[not found]"
        status = "FOUND" if is_found else "NOT FOUND"
        sources = str(r["sources"]) if r["sources"] else "[]"
        print(f"  {r['id']:<4} {sources:<20} {status:<12} {answer_preview}")

    print()
    print("=" * 70)
    print("  All 5 questions processed successfully!")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 5: Save results to JSON for reference
    # ----------------------------------------------------------
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
