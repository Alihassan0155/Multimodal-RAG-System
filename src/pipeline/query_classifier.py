# src/pipeline/query_classifier.py

import json
import re
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL          = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

_classify_cache: dict = {}


def _groq() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Query Intent Classification
#
# Not all queries need the same retrieval strategy.
# Sending all queries through identical retrieval parameters wastes
# precision on aggregation queries and over-retrieves on simple lookups.
#
# Four intent types with distinct retrieval needs:
#
# LOOKUP
#   "what does MIC stand for?"
#   User wants one specific fact from one specific place.
#   → top_k=3, tight threshold, precise retrieval
#
# AGGREGATION
#   "list all key terms defined in the document"
#   User wants synthesised information across the whole document.
#   → top_k=10, loose threshold, broad retrieval, synthesis instruction
#
# COMPARISON
#   "what is the difference between GNN and simple ML models?"
#   User wants two or more concepts compared side by side.
#   → top_k=6, moderate threshold, retrieve both topics
#
# BOOLEAN
#   "does the system use plasmids as features?"
#   User wants yes/no confirmation with evidence.
#   → top_k=3, tight threshold, yes/no answer format
#
# WHERE ELSE IS INTENT CLASSIFICATION USED:
#   Search engines (Google classifies navigational vs informational vs
#   transactional queries), dialogue systems, question answering systems.
#   In production RAG: Cohere, LlamaIndex, and LangChain all implement
#   some form of query routing based on intent.
# ─────────────────────────────────────────────────────────────────────────────

# Retrieval config per intent type
INTENT_CONFIGS = {
    "lookup": {
        "top_k":          3,
        "threshold":      -8.0,
        "expand_queries": True,
        "description":    "Precise fact retrieval",
        "llm_instruction": (
            "Answer the specific question directly and concisely. "
            "Focus on the single most relevant fact."
        ),
    },
    "explanation": {
        "top_k":          5,
        "threshold":      -8.0,
        "expand_queries": True,
        "description":    "Conceptual explanation requiring multiple supporting chunks",
        "llm_instruction": (
            "Explain the concept clearly and thoroughly. "
            "Use step-by-step reasoning where applicable. "
            "Synthesise information across all provided excerpts."
        ),
    },
    "aggregation": {
        "top_k":          10,
        "threshold":      -8.0,
        "expand_queries": True,
        "description":    "Broad retrieval + synthesis",
        "llm_instruction": (
            "Synthesise information across ALL provided excerpts. "
            "Compile a complete list or summary — do not stop at the first mention. "
            "If the same concept appears in multiple excerpts, merge them."
        ),
    },
    "comparison": {
        "top_k":          6,
        "threshold":      -5.0,
        "expand_queries": True,
        "description":    "Multi-topic retrieval for comparison",
        "llm_instruction": (
            "Compare the concepts directly. "
            "Structure your answer with clear contrast — similarities and differences. "
            "Use evidence from multiple excerpts."
        ),
    },
    "boolean": {
        "top_k":          3,
        "threshold":      -5.0,
        "expand_queries": False,  # boolean queries are already precise
        "description":    "Yes/no with supporting evidence",
        "llm_instruction": (
            "Answer with YES or NO first, then provide the supporting evidence "
            "from the excerpts. Be definitive — do not hedge if the evidence is clear."
        ),
    },
}


CLASSIFIER_PROMPT = """You are a query intent classifier for a RAG (Retrieval-Augmented Generation) system.

## Your task
Classify the user query into exactly one of these intent types:

**lookup**
The user wants one specific fact, value, or short definition (answerable in 1-2 sentences).
Examples: "what does MIC stand for?", "what is the accuracy?", "define plasmid"

**explanation**
The user wants a conceptual explanation of how something works, what something is in depth,
or an understanding of a mechanism/architecture. These questions need multiple supporting chunks.
Examples: "what is self-attention?", "how does the transformer work?", "explain backpropagation",
"what is BERT and how does it work?", "how does attention differ from convolution?"

**aggregation**
The user wants a compiled list, summary, or synthesis across the whole document.
Examples: "list all key terms", "summarise the methods", "what are all the features used?"

**comparison**
The user wants two or more concepts compared or contrasted.
Examples: "difference between GNN and CNN", "compare lattice vs stream mode"

**boolean**
The user wants a yes/no answer with supporting evidence.
Examples: "does the system use plasmids?", "is GNN used for prediction?", "are k-mers included?"

## Classification rules
- Choose the SINGLE best matching intent
- Key rule: if the query asks "what is X AND how does it work" or "explain X" → explanation (not lookup)
- When in doubt between lookup and explanation: if a thorough answer needs more than 2 sentences → explanation
- When in doubt between lookup and aggregation: if the query contains "all", "list", "every" → aggregation
- When in doubt between lookup and boolean: if the query can be answered yes/no → boolean
- Output ONLY valid JSON, no explanation

## Output format
{{"intent": "lookup|explanation|aggregation|comparison|boolean", "confidence": 0.0-1.0, "reason": "one sentence"}}

## Examples
Query: "what does MIC stand for?"
Output: {{"intent": "lookup", "confidence": 0.98, "reason": "Asks for definition of a specific acronym"}}

Query: "what are transformers and how does self-attention work?"
Output: {{"intent": "explanation", "confidence": 0.97, "reason": "Asks for deep conceptual explanation of an architecture and its core mechanism"}}

Query: "how does scaled dot-product attention compute scores?"
Output: {{"intent": "explanation", "confidence": 0.95, "reason": "Asks how a mechanism works — needs multiple excerpts covering the formula and process"}}

Query: "list all key terms defined in the document"
Output: {{"intent": "aggregation", "confidence": 0.97, "reason": "Asks to list ALL terms across the document"}}

Query: "what is the difference between GNN and simple ML models?"
Output: {{"intent": "comparison", "confidence": 0.95, "reason": "Explicitly asks for a difference between two approaches"}}

Query: "does the system use graph neural networks?"
Output: {{"intent": "boolean", "confidence": 0.96, "reason": "Yes/no question about system feature"}}

---

Query: "{query}"
Output:"""


def classify_query(query: str) -> dict:
    """
    Classifies query intent and returns retrieval configuration.

    Returns:
    {
        "intent":         "lookup" | "aggregation" | "comparison" | "boolean"
        "confidence":     0.0 - 1.0
        "reason":         "why this intent was chosen"
        "top_k":          int
        "threshold":      float
        "expand_queries": bool
        "llm_instruction": str  ← injected into answering prompt
    }

    🧠 CONCEPT: why return llm_instruction from classifier
    The answering LLM needs different instructions depending on intent.
    An aggregation answer requires "synthesise across all excerpts".
    A boolean answer requires "answer YES/NO first".
    Generating these instructions from the classifier keeps the logic
    centralised — one place controls both retrieval AND answer style.
    """
    print(f"  [Classifier] Classifying intent for: '{query[:60]}'")

    if query in _classify_cache:
        print("  [Classifier] Cache hit")
        return _classify_cache[query]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a query intent classifier. "
                "Output ONLY valid JSON. No markdown, no explanation."
            )
        },
        {
            "role": "user",
            "content": CLASSIFIER_PROMPT.format(query=query)
        }
    ]

    raw = None
    delay = 5
    for attempt in range(3):
        try:
            raw = _groq().chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            ).choices[0].message.content.strip()
            break
        except Exception as e:
            if "429" not in str(e):
                raise
            if attempt < 2:
                print(f"  [Classifier Retry {attempt + 1}] Rate limit — waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  [Classifier Fallback] Switching to {FALLBACK_MODEL}")
                raw = _groq().chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=100,
                ).choices[0].message.content.strip()

    try:
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

        data       = json.loads(raw)
        intent     = data.get("intent", "lookup")
        confidence = data.get("confidence", 0.8)
        reason     = data.get("reason", "")

        # Validate intent
        if intent not in INTENT_CONFIGS:
            print(f"  [Classifier] Unknown intent '{intent}' — defaulting to explanation")
            intent = "explanation"

        config = INTENT_CONFIGS[intent].copy()
        config["intent"]     = intent
        config["confidence"] = confidence
        config["reason"]     = reason

        print(f"  [Classifier] Intent: {intent} "
              f"(confidence: {confidence:.2f}) — {reason}")

        _classify_cache[query] = config
        return config

    except Exception as e:
        print(f"  [Classifier] Failed: {e} — defaulting to lookup config")
        config = INTENT_CONFIGS["lookup"].copy()
        config["intent"]     = "lookup"
        config["confidence"] = 0.5
        config["reason"]     = "fallback due to classifier error"
        return config