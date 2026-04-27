"""
RAG Evaluation Script
Calls the pipeline directly (no HTTP server needed).
Run: python evaluate.py
"""

import re
import sys
import os
from pathlib import Path
from typing import List, Dict

# ── Ensure project root is on the path ───────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Load env vars (GROQ_API_KEY etc.) ────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Pipeline imports (mirrors _run_query in api.py) ──────────────────────────
from src.pipeline.query_classifier import classify_query
from src.pipeline.answerer import expand_query, answer
from src.retrieval.retriever import HybridRetriever

_retriever = HybridRetriever(top_candidates=20)


# ==============================
# 🔌 PIPELINE ADAPTER
# ==============================
def ask_model(question: str, history: List[str] = None) -> str:
    """
    Calls the RAG pipeline directly.
    history: list of "Q: ...\nA: ..." strings (eval harness format).
    """
    # Convert eval-harness history strings → pipeline dicts
    history_dicts: list[dict] = []
    for entry in (history or []):
        parts = entry.split("\nA: ", 1)
        if len(parts) == 2:
            q_part = parts[0].removeprefix("Q: ").strip()
            a_part = parts[1].strip()
            history_dicts.append({"role": "user",      "content": q_part})
            history_dicts.append({"role": "assistant",  "content": a_part})

    intent_config   = classify_query(question)
    intent          = intent_config["intent"]
    top_k           = intent_config["top_k"]
    expand_enabled  = intent_config["expand_queries"]
    threshold       = intent_config["threshold"]

    expanded_queries = expand_query(question) if expand_enabled else [question]

    chunks = _retriever.retrieve_multi(
        queries=expanded_queries,
        original_query=question,
        top_k=top_k,
        intent_threshold=threshold,
    )

    if not chunks:
        return "This information is not in the document."

    return answer(query=question, chunks=chunks, intent=intent, history=history_dicts)


# ==============================
# 🧠 UTILS
# ==============================
def contains_formula(text: str) -> bool:
    return "softmax" in text.lower() or ("q" in text.lower() and "k" in text.lower())

def has_citation(text: str) -> bool:
    return bool(re.search(r"\[.*?\|\s*Page\s*\d+.*?\]", text))

def is_concise(text: str, max_words: int = 300) -> bool:
    return len(text.split()) <= max_words

def has_redundancy(text: str) -> bool:
    sentences = text.split(". ")
    return len(sentences) != len(set(sentences))

def contains_reference_phrase(text: str) -> bool:
    phrases = ["as explained earlier", "building on", "as discussed"]
    return any(p in text.lower() for p in phrases)

def repeats_definition(prev: str, curr: str) -> bool:
    return len(set(prev.split()) & set(curr.split())) > 20


# ==============================
# 🧪 TEST RUNNER
# ==============================
def _is_rate_limited(text: str) -> bool:
    return "429" in text or "rate limit" in text.lower() or "Error code" in text


def run_test_case(name: str, steps: List[str], checks: Dict):
    print(f"\n=== TEST: {name} ===")

    history: List[str] = []
    responses: List[str] = []
    rate_limited = False

    for i, q in enumerate(steps):
        response = ask_model(q, history)
        responses.append(response)
        history.append(f"Q: {q}\nA: {response}")

        print(f"\nQ{i+1}: {q}")
        print(f"A{i+1}: {response[:200]}...\n")

        if _is_rate_limited(response):
            rate_limited = True

    if rate_limited:
        print("  ⚠️  Skipped — rate limit hit during this test")
        return None, None

    score = 0
    total = len(checks)

    for check_name, check_fn in checks.items():
        try:
            if check_fn(responses):
                print(f"  ✅ {check_name}")
                score += 1
            else:
                print(f"  ❌ {check_name}")
        except Exception as e:
            print(f"  ⚠️  {check_name} ERROR: {e}")

    print(f"\n  Score: {score}/{total}")
    return score, total


# ==============================
# 📊 TEST SUITE
# ==============================
def run_all_tests():
    results = []

    results.append(run_test_case(
        "A1: Self-Attention Definition",
        ["What is self-attention?"],
        {
            "Has formula":    lambda r: contains_formula(r[0]),
            "Has citation":   lambda r: has_citation(r[0]),
            "Concise":        lambda r: is_concise(r[0]),
            "No redundancy":  lambda r: not has_redundancy(r[0]),
        }
    ))

    results.append(run_test_case(
        "C1: Memory Follow-up",
        [
            "What is self-attention?",
            "How is it used in Transformers?",
        ],
        {
            "No redefinition":      lambda r: not repeats_definition(r[0], r[1]),
            "Uses reference phrase": lambda r: contains_reference_phrase(r[1]),
        }
    ))

    results.append(run_test_case(
        "C2: Pronoun Resolution",
        [
            "Explain multi-head attention",
            "Why is it better than single-head?",
        ],
        {
            "No repetition": lambda r: not repeats_definition(r[0], r[1]),
            "Concise":       lambda r: is_concise(r[1]),
        }
    ))

    results.append(run_test_case(
        "B1: Structure Consistency",
        ["What is multi-head attention?"],
        {
            "Concise":       lambda r: is_concise(r[0]),
            "No redundancy": lambda r: not has_redundancy(r[0]),
        }
    ))

    results.append(run_test_case(
        "D1: Citation Format",
        ["Explain Transformer architecture"],
        {
            "Has citation": lambda r: has_citation(r[0]),
        }
    ))

    results.append(run_test_case(
        "H1: Out of Scope",
        ["Who won FIFA World Cup 2022?"],
        {
            "No hallucination": lambda r: (
                "not in document" in r[0].lower()
                or "outside scope" in r[0].lower()
                or "not available" in r[0].lower()
                or "no relevant" in r[0].lower()
            ),
        }
    ))

    valid   = [(s, t) for s, t in results if s is not None]
    skipped = sum(1 for s, _ in results if s is None)

    total_score = sum(s for s, _ in valid)
    total_tests = sum(t for _, t in valid)

    print("\n==============================")
    print(f"  FINAL SCORE: {total_score}/{total_tests} (valid tests)")
    if skipped:
        print(f"  SKIPPED:     {skipped} test(s) — rate limit")
    print("==============================\n")


if __name__ == "__main__":
    run_all_tests()
