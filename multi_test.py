"""
============================================================
multi_test.py — Multi-Test Suite for Project 7 RAG System
============================================================
PURPOSE:
    Conducts extensive multiple testing of the RAG system to
    verify correctness, resilience, and adherence to rules.

TEST SUITES:
    1. Core Assessment Questions (Q1 - Q5)
    2. Question Paraphrasing / Variations
    3. Negative / Out-of-Domain Guardrails (Strict Grounding)
    4. Exact Source Page Attribution Verification
    5. Schema & Type Consistency (API Contract)

HOW TO RUN:
    python3 multi_test.py
============================================================
"""

import sys
import time
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)

from rag_engine import answer


def run_test(name: str, test_fn):
    """Helper function to execute a test and report pass/fail."""
    try:
        test_fn()
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: {type(e).__name__} - {e}")
        return False


def main():
    print("=" * 75)
    print("  PROJECT 7 RAG SYSTEM — COMPREHENSIVE MULTI-TEST VERIFICATION")
    print("=" * 75)
    print()

    total_tests = 0
    passed_tests = 0

    # ==============================================================
    # SUITE 1: Primary Assessment Questions (Q1 to Q5)
    # ==============================================================
    print("--- SUITE 1: Primary Assessment Questions ---")

    def test_q1_primary():
        q = "What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?"
        res = answer(q)
        assert isinstance(res, dict), "Result must be a dict"
        assert "3e-4" in res["answer"] or "0.0003" in res["answer"], "Missing learning rate 3e-4"
        assert "4,000" in res["answer"], "Missing 4,000 warmup steps"
        assert "0.9" in res["answer"] and "0.95" in res["answer"], "Missing Adam beta values"
        assert res["sources"] == [16], f"Expected source [16], got {res['sources']}"

    def test_q2_primary():
        q = "Who is the current Chief Executive Officer of OpenAI?"
        res = answer(q)
        assert res["answer"] == "[not found]", f"Expected '[not found]', got '{res['answer']}'"
        assert res["sources"] == [], f"Expected sources [], got {res['sources']}"

    def test_q3_primary():
        q = "By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?"
        res = answer(q)
        assert "14.6%" in res["answer"], "Missing 14.6% improvement"
        assert "61.2%" in res["answer"] and "75.8%" in res["answer"], "Missing baseline/hybrid EM numbers"
        assert res["sources"] == [31], f"Expected source [31], got {res['sources']}"

    def test_q4_primary():
        q = "What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?"
        res = answer(q)
        assert "128 NVIDIA H100" in res["answer"], "Missing 128 NVIDIA H100 GPUs"
        assert "18.5 days" in res["answer"], "Missing 18.5 days duration"
        assert res["sources"] == [45], f"Expected source [45], got {res['sources']}"

    def test_q5_primary():
        q = "What was the worldwide box-office revenue of the film Oppenheimer in 2023?"
        res = answer(q)
        assert res["answer"] == "[not found]", f"Expected '[not found]', got '{res['answer']}'"
        assert res["sources"] == [], f"Expected sources [], got {res['sources']}"

    suite_1 = [
        ("Q1 Primary Assessment: Pre-training Hyperparameters", test_q1_primary),
        ("Q2 Primary Assessment: OpenAI CEO (Out-of-domain)", test_q2_primary),
        ("Q3 Primary Assessment: Hybrid Retrieval EM Accuracy Gain", test_q3_primary),
        ("Q4 Primary Assessment: GPU Cluster & Training Duration", test_q4_primary),
        ("Q5 Primary Assessment: Oppenheimer Box Office (Out-of-domain)", test_q5_primary),
    ]

    for name, fn in suite_1:
        total_tests += 1
        if run_test(name, fn):
            passed_tests += 1

    print()

    # ==============================================================
    # SUITE 2: Semantic Paraphrase & Robustness Variations
    # ==============================================================
    print("--- SUITE 2: Semantic Paraphrase & Robustness Variations ---")

    def test_q1_variation_a():
        q = "What was the learning rate and Adam beta values used for model pretraining?"
        res = answer(q)
        assert "3e-4" in res["answer"] or "0.0003" in res["answer"]
        assert res["sources"] == [16]

    def test_q1_variation_b():
        q = "What optimizer beta values and warmup steps are reported for pre-training in Section 4?"
        res = answer(q)
        assert "0.9" in res["answer"] and "0.95" in res["answer"]
        assert "4,000" in res["answer"]
        assert res["sources"] == [16]

    def test_q3_variation_a():
        q = "How much did hybrid retrieval improve the exact match accuracy over dense baseline?"
        res = answer(q)
        assert "14.6%" in res["answer"]
        assert res["sources"] == [31]

    def test_q3_variation_b():
        q = "What was the EM accuracy gain on the benchmark dataset from the proposed hybrid retrieval method?"
        res = answer(q)
        assert "14.6%" in res["answer"]
        assert res["sources"] == [31]

    def test_q4_variation_a():
        q = "How many GPUs and what model were utilized for training the 13B model?"
        res = answer(q)
        assert "128 NVIDIA H100" in res["answer"]
        assert res["sources"] == [45]

    def test_q4_variation_b():
        q = "What was the training duration in days required for the 13-billion-parameter model on the GPU cluster?"
        res = answer(q)
        assert "18.5 days" in res["answer"]
        assert res["sources"] == [45]

    suite_2 = [
        ("Q1 Paraphrase A (Learning rate & Adam betas)", test_q1_variation_a),
        ("Q1 Paraphrase B (Optimizer betas & warmup in Section 4)", test_q1_variation_b),
        ("Q3 Paraphrase A (Hybrid retrieval EM accuracy improvement)", test_q3_variation_a),
        ("Q3 Paraphrase B (EM benchmark gain percentage)", test_q3_variation_b),
        ("Q4 Paraphrase A (Number and model of GPUs for 13B)", test_q4_variation_a),
        ("Q4 Paraphrase B (Training duration in days for 13B)", test_q4_variation_b),
    ]

    for name, fn in suite_2:
        total_tests += 1
        if run_test(name, fn):
            passed_tests += 1

    print()

    # ==============================================================
    # SUITE 3: Negative / Out-of-Domain Guardrails (Strict Grounding)
    # ==============================================================
    print("--- SUITE 3: Negative & Out-of-Domain Guardrails (Strict Grounding) ---")

    negative_questions = [
        ("Who is Sam Altman?", "Negative: Sam Altman"),
        ("What is the capital city of France?", "Negative: Geography"),
        ("What was the worldwide gross of Barbie movie in 2023?", "Negative: Barbie movie"),
        ("Who is the CEO of Google or Microsoft?", "Negative: Tech CEOs"),
        ("What is the speed of light in vacuum?", "Negative: Physics constant"),
    ]

    for n_q, label in negative_questions:
        def make_neg_test(query):
            def neg_test():
                res = answer(query)
                assert res["answer"] == "[not found]", f"Expected '[not found]', got '{res['answer']}'"
                assert res["sources"] == [], f"Expected sources [], got '{res['sources']}'"
            return neg_test

        total_tests += 1
        if run_test(label, make_neg_test(n_q)):
            passed_tests += 1

    print()

    # ==============================================================
    # SUITE 4: API Contract & Schema Compliance
    # ==============================================================
    print("--- SUITE 4: API Contract & Schema Compliance ---")

    def test_schema_compliance():
        res = answer("What exact learning rate was used in Section 4?")
        assert isinstance(res, dict), "Result must be a dict"
        assert "answer" in res and "sources" in res, "Result must contain 'answer' and 'sources'"
        assert isinstance(res["answer"], str), "'answer' must be str"
        assert isinstance(res["sources"], list), "'sources' must be list"
        for s in res["sources"]:
            assert isinstance(s, (int, str)), "Sources items must be int page numbers or str chunk IDs"

    def test_speed_and_persistence():
        start = time.time()
        for _ in range(5):
            answer("Who is the CEO of OpenAI?")
        elapsed = time.time() - start
        avg_time = elapsed / 5.0
        assert avg_time < 1.0, f"Average query took {avg_time:.2f}s, expected < 1.0s with persisted DB"

    suite_4 = [
        ("Response Schema Contract Verification", test_schema_compliance),
        ("Database Persistence & Query Latency (< 1.0s)", test_speed_and_persistence),
    ]

    for name, fn in suite_4:
        total_tests += 1
        if run_test(name, fn):
            passed_tests += 1

    print()
    print("=" * 75)
    print(f"  MULTI-TEST SUMMARY: {passed_tests} / {total_tests} TESTS PASSED ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 75)
    print()

    if passed_tests == total_tests:
        print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"  WARNING: {total_tests - passed_tests} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
