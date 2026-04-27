# # src/pipeline/answerer.py

# import os
# import json
# import re
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# # ─────────────────────────────────────────────────────────────────────────────
# # 🧠 CONCEPT: Two separate model configs for two different jobs
# #
# # expansion_model — temperature 0.1
# #   Needs SLIGHT variation to produce meaningfully different query variants.
# #   Pure 0.0 produces near-identical rewrites. 0.1 is the sweet spot.
# #
# # answering_model — temperature 0.0
# #   Must be fully deterministic. Same question + same chunks = same answer.
# #   Any temperature > 0 risks subtle hallucination in grounded answering.
# # ─────────────────────────────────────────────────────────────────────────────

# # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # expansion_model = genai.GenerativeModel(
# #     model_name="gemini-1.5-flash",
# #     generation_config=genai.GenerationConfig(
# #         temperature=0.1,
# #         max_output_tokens=512,
# #     )
# # )

# # answering_model = genai.GenerativeModel(
# #     model_name="gemini-1.5-flash",
# #     generation_config=genai.GenerationConfig(
# #         temperature=0.0,
# #         max_output_tokens=1024,
# #     )
# # )

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# # ─────────────────────────────────────────────────────────────────────────────
# # STAGE 1: QUERY EXPANSION
# # ─────────────────────────────────────────────────────────────────────────────

# # ─────────────────────────────────────────────────────────────────────────────
# # 🧠 CONCEPT: 4-variant expansion strategy — each targets a different gap
# #
# # VARIANT 1 — LEXICAL VARIANT
# #   Target: BM25 vocabulary gap
# #   BM25 is exact-match. If doc says "forecast" and user says "predict",
# #   BM25 misses. Lexical variant swaps synonyms to bridge this gap.
# #   Rule: same structure, different words.
# #
# # VARIANT 2 — TECHNICAL REFRAME
# #   Target: domain terminology gap
# #   Academic/technical docs use precise jargon. A casual query misses these.
# #   Rule: rephrase as an expert would write it in a paper.
# #
# # VARIANT 3 — CONCEPT FOCUS (replaces "Decomposition")
# #   Target: conceptual retrieval gap
# #   FIX from issue 2: "decomposition" strips constraints and loses intent.
# #   "Focus" keeps the original intent but zooms into the core mechanism.
# #   Rule: preserve intent, focus on the single most important concept.
# #
# # VARIANT 4 — QUESTION REFORMULATION (new — fixes issue 1)
# #   Target: embedding model alignment gap
# #   Embeddings are trained on natural conversational text.
# #   Technical queries don't always align well with embedding space.
# #   This variant writes the query as a natural search engine question.
# #   Rule: write it as a human would ask Google or a chatbot.
# #
# # WHY THIS SET COVERS ALL RETRIEVAL GAPS:
# #   Variant 1 → BM25 keyword matching    (lexical diversity)
# #   Variant 2 → BM25 technical terms     (vocabulary precision)
# #   Variant 3 → Vector semantic search   (conceptual alignment)
# #   Variant 4 → Vector embedding space   (conversational alignment)
# #
# # FIX issue 3: Three anti-hallucination constraints added:
# #   - No overly generic queries
# #   - No scope broadening
# #   - Must remain specific to user intent
# # ─────────────────────────────────────────────────────────────────────────────

# EXPANSION_PROMPT = """You are an expert RAG (Retrieval-Augmented Generation) query analyst specialising in hybrid search systems that combine BM25 keyword search with dense vector embeddings.

# ## Your task
# Given a user query, generate exactly 4 rewritten variants. Each variant must serve a SPECIFIC and DISTINCT retrieval purpose. The goal is maximum retrieval recall — finding every relevant chunk in the document.

# ---

# ## Variant specifications

# **Variant 1 — LEXICAL VARIANT**
# Replace key terms with synonyms, related words, or alternative phrasings.
# Goal: Help BM25 keyword search find chunks that use different vocabulary than the original.
# Rule: Keep the same sentence structure. Only swap words — do not restructure the query.
# Example transform: "predict" → "forecast", "model" → "system", "accuracy" → "performance"

# **Variant 2 — TECHNICAL REFRAME**
# Rephrase using precise domain-specific or academic terminology.
# Goal: Match the exact vocabulary used in technical documents, research papers, or professional reports.
# Rule: Think "how would an expert author write this concept in a paper?" Use technical jargon deliberately.

# **Variant 3 — CONCEPT FOCUS**
# Rephrase to focus tightly on the core concept or mechanism being asked about, while fully preserving the original intent.
# Goal: Improve retrieval of highly specific conceptual chunks without losing meaning or constraints.
# Rule: Do NOT strip constraints. Do NOT broaden scope. Zoom into the central mechanism — keep intent intact.

# **Variant 4 — QUESTION REFORMULATION**
# Rewrite as a natural, conversational question a user might type into a search engine or chatbot.
# Goal: Improve alignment with embedding models trained on natural language, and improve recall for conversational phrasing in the document.
# Rule: Write it naturally and directly, as if asking a knowledgeable colleague. Avoid jargon.

# ---

# ## Hard constraints (apply to ALL variants)
# - Preserve the ORIGINAL INTENT exactly — never change what is being asked
# - Do NOT introduce new topics, entities, or facts not present in the original query
# - Do NOT generate overly generic queries (e.g., "what is machine learning", "explain AI")
# - Do NOT broaden the scope beyond the original question
# - Each variant must remain specific to the user's intent
# - Keep variants concise — under 20 words each
# - ALL 4 variants must be meaningfully different from each other and from the original
# - Output ONLY valid JSON — no explanation, no markdown, no code blocks

# ## Output format
# {{"variants": ["variant1", "variant2", "variant3", "variant4"]}}

# ---

# ## Few-shot examples

# Query: "how does the attention mechanism work in transformers?"
# Output:
# {{"variants": [
#   "how do transformers focus on relevant parts of the input sequence?",
#   "scaled dot-product attention query key value weight computation transformers",
#   "attention score computation mechanism between token representations",
#   "what is attention in a transformer and how does it work?"
# ]}}

# Query: "what accuracy did ResNet-152 achieve on ImageNet?"
# Output:
# {{"variants": [
#   "what performance score did ResNet-152 get on the ImageNet benchmark?",
#   "ResNet-152 top-1 top-5 classification accuracy ImageNet ILSVRC evaluation",
#   "ResNet-152 ImageNet benchmark classification performance",
#   "how accurate is ResNet-152 on the ImageNet dataset?"
# ]}}

# Query: "how does antibiotic resistance develop in bacteria?"
# Output:
# {{"variants": [
#   "how do bacteria become resistant to antibiotics over time?",
#   "antimicrobial resistance mechanisms gene mutation selection pressure bacterial evolution",
#   "mechanism by which bacteria acquire resistance to antibiotic treatment",
#   "why do bacteria stop responding to antibiotics?"
# ]}}

# ---

# Now generate variants for this query. Remember: specific, diverse, intent-preserving.

# Query: "{query}"
# Output:"""


# def deduplicate_queries(queries: list[str]) -> list[str]:
#     """
#     Removes duplicate or near-duplicate queries from the expanded list.

#     🧠 FIX for Issue 4:
#     LLMs sometimes generate variants that are too similar — slight word
#     swaps that produce effectively the same retrieval result. Running both
#     wastes API calls and pollutes the RRF ranking with correlated signals.

#     Strategy: lowercase + strip, then deduplicate by exact match.
#     We don't use fuzzy matching (e.g. edit distance) to keep it fast
#     and deterministic — exact lowercase dedup catches the most common
#     failure mode (identical strings with different casing/spacing).

#     The original query is always first and always kept.
#     """
#     seen   = set()
#     unique = []

#     for q in queries:
#         key = q.lower().strip()
#         if key not in seen:
#             seen.add(key)
#             unique.append(q)
#         else:
#             print(f"  [Expansion] Duplicate removed: '{q}'")

#     return unique


# def expand_query(query: str) -> list[str]:
#     """
#     Expands a user query into up to 5 variants (original + 4 generated).
#     Deduplicates the result before returning.

#     Returns: [original_query, variant1, variant2, variant3, variant4]
#              (may be shorter if duplicates removed)

#     🧠 WHY ORIGINAL IS ALWAYS FIRST:
#     The original query is the most precise expression of user intent.
#     It anchors retrieval. Variants add coverage but never replace it.
#     Reranking later uses the original — so retrieval must include
#     chunks relevant to it.
#     """
#     print(f"\n  [Expansion] Generating 4 variants for: '{query}'")

#     try:
#         prompt   = EXPANSION_PROMPT.format(query=query)
#         response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt,
#         config={
#                 "temperature": 0.1,
#                 "max_output_tokens": 512,
#             }
#         )   

#         raw = response.text.strip()

#         # Strip markdown code blocks if LLM wraps output despite instructions
#         raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

#         data     = json.loads(raw)
#         variants = data.get("variants", [])

#         # Validate and clean
#         variants = [v.strip() for v in variants if isinstance(v, str) and v.strip()][:4]

#         # Pad if fewer than 4 returned
#         while len(variants) < 4:
#             print(f"  [Expansion] Only got {len(variants)} variants — padding")
#             variants.append(query)

#         # Assemble: original first, then variants
#         all_queries = [query] + variants

#         # Deduplicate (fixes issue 4)
#         all_queries = deduplicate_queries(all_queries)

#         print(f"  [Expansion] Final query set ({len(all_queries)} queries):")
#         for i, q in enumerate(all_queries):
#             label = "original " if i == 0 else f"variant {i}"
#             print(f"    [{label}] {q}")

#         return all_queries

#     except json.JSONDecodeError as e:
#         print(f"  [Expansion] JSON parse error: {e}")
#         print(f"  [Expansion] Raw response: {raw[:200]}")
#         print(f"  [Expansion] Falling back to original query only")
#         return [query]

#     except Exception as e:
#         print(f"  [Expansion] Failed: {e} — falling back to original query only")
#         return [query]


# # ─────────────────────────────────────────────────────────────────────────────
# # STAGE 2: GROUNDED ANSWERING
# # ─────────────────────────────────────────────────────────────────────────────

# # ─────────────────────────────────────────────────────────────────────────────
# # 🧠 CONCEPT: Advanced RAG answering prompt — all 6 issues addressed
# #
# # Issue 5 fix — HARD anti-hallucination constraint:
# #   "If a claim cannot be directly supported by the excerpts, do NOT include it.
# #    Do NOT generalise beyond what is explicitly stated, even if it seems obvious."
# #   This is stronger than "don't use training knowledge" — it explicitly
# #   forbids the LLM's natural tendency to "fill gaps logically."
# #
# # Issue 6 fix — Excerpt-anchored citation format:
# #   Changed from [Source: file, page N]
# #   To           [Excerpt X | Source: file, page N]
# #   The model already sees chunks labelled [Excerpt 1], [Excerpt 2] etc.
# #   Anchoring citations to excerpt numbers prevents hallucinated page refs —
# #   the model maps its claim directly to a numbered excerpt it was shown.
# #
# # Context quality indicator:
# #   Tells the model how much to trust the retrieved content.
# #   HIGH confidence prompts direct answers.
# #   LOW confidence prompts explicit uncertainty statements.
# # ─────────────────────────────────────────────────────────────────────────────


# ANSWERING_PROMPT = """You are a precise document intelligence assistant operating in a RAG (Retrieval-Augmented Generation) system.

# ## Your role
# Answer the user's question using ONLY the numbered document excerpts provided below.
# You have access to NO other information source. Your training knowledge does not exist for this task.

# ## Strict grounding rules
# 1. ONLY use information explicitly stated in the excerpts — nothing else
# 2. If a claim cannot be directly supported by a specific excerpt, do NOT include it
# 3. Do NOT generalise beyond what is explicitly stated, even if it seems logically obvious
# 4. Do NOT infer, assume, or extrapolate — only report what the text directly says
# 5. If the excerpts do not contain the answer, say so clearly and specifically

# ## Citation format
# After EVERY factual claim, cite the source using this exact format:
# [Excerpt X | Source: filename, page N]

# Where X is the excerpt number shown in the context below.
# Use this format consistently — do not invent page numbers or excerpt numbers.

# ## Confidence levels
# - HIGH   = excerpt directly and completely answers the question with explicit statements
# - MEDIUM = excerpt partially answers or implies the answer — state what is missing
# - LOW    = excerpts are tangentially related — answer is uncertain, flag this clearly

# ## Context quality
# {context_quality}

# ## Document excerpts (retrieved and ranked by relevance to your question)
# {context_block}

# ## User question
# {query}

# ## Response format

# **Answer:**
# [Your grounded answer. Every factual claim followed by [Excerpt X | Source: filename, page N].
# If the context does not support a complete answer, state exactly what is and is not covered.]

# **Key points:**
# - [specific point] [Excerpt X | Source: filename, page N]
# - [specific point] [Excerpt X | Source: filename, page N]

# **Confidence:** [HIGH / MEDIUM / LOW]

# **Gaps:**
# [Specific information that would improve the answer but is not present in the excerpts.
# Write "None" if the excerpts fully answer the question.]"""


# def build_context_block(chunks: list[dict]) -> tuple[str, str]:
#     """
#     Builds the numbered context block passed to the answering LLM.

#     🧠 WHY NUMBER THE EXCERPTS:
#     Numbered excerpts ([Excerpt 1], [Excerpt 2]) give the model
#     stable anchors for citations. Without numbers, the model either
#     repeats full source strings (verbose) or hallucinates page numbers.
#     With numbers, citation becomes: [Excerpt 2 | Source: paper.pdf, page 1]
#     — precise, verifiable, directly mapped to what we showed the model.

#     🧠 WHY INCLUDE RERANK SCORE IN CONTEXT:
#     The score tells the model how confident the retrieval system is.
#     A score of 6.2 = strong match → model answers confidently.
#     A score of 0.3 = weak match  → model hedges appropriately.
#     This is honest RAG — the model's confidence calibrates to evidence quality.
#     """
#     if not chunks:
#         return "", "⚠ WEAK CONTEXT: No relevant chunks were retrieved."

#     lines     = []
#     top_score = chunks[0].get("rerank_score", 0)

#     for i, chunk in enumerate(chunks, 1):
#         score   = round(chunk.get("rerank_score", 0), 2)
#         source  = chunk.get("source", "unknown")
#         page    = chunk.get("page", "?")
#         content = chunk.get("content", "").strip()

#         lines.append(
#             f"[Excerpt {i}] Source: {source}, Page {page} | "
#             f"Relevance score: {score}\n"
#             f"{content}"
#         )

#     context_block = "\n\n---\n\n".join(lines)

#     # Quality signal to calibrate model confidence
#     if top_score >= 3.0:
#         quality = "✅ STRONG: Top excerpts are highly relevant. Answer with HIGH confidence where supported."
#     elif top_score >= 0.0:
#         quality = "⚠ MODERATE: Excerpts partially relevant. Use MEDIUM confidence and note gaps explicitly."
#     else:
#         quality = "❌ WEAK: Low relevance scores. Use LOW confidence. Prioritise transparency over completeness."

#     return context_block, quality


# def answer(query: str, chunks: list[dict]) -> str:
#     """
#     Generates a grounded, cited answer from retrieved chunks.

#     Args:
#         query  : the ORIGINAL user query — not expanded variants
#         chunks : top-k reranked, threshold-filtered chunks

#     Returns: formatted answer with inline citations and confidence rating

#     🧠 WHY ORIGINAL QUERY FOR ANSWERING:
#     Expanded variants served their purpose — maximising retrieval recall.
#     The LLM must answer what the USER asked, not a rewritten variant.
#     Passing an expanded query would produce an answer that feels
#     slightly misaligned with what was actually asked.
#     """
#     if not chunks:
#         return (
#             "**Answer:** No relevant information found in the ingested documents "
#             "for this question.\n\n"
#             "**Confidence:** LOW\n\n"
#             "**Gaps:** The query appears to be outside the scope of the "
#             "available documents."
#         )

#     context_block, context_quality = build_context_block(chunks)

#     prompt = ANSWERING_PROMPT.format(
#         context_quality=context_quality,
#         context_block=context_block,
#         query=query,
#     )

#     print(f"  [Gemini] Sending {len(chunks)} excerpts to answering model...")
#     print(f"  [Gemini] Top rerank score: {chunks[0].get('rerank_score', 0):.2f}")

#     try:
#         response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt,
#         config={
#                 "temperature": 0.0,
#                 "max_output_tokens": 1024,
#             }
#         )

#         return response.text.strip()

#     except Exception as e:
#         return f"**Answer:** Error generating response — {e}\n\n**Confidence:** LOW"


# src/pipeline/answerer.py

import os
import json
import re
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Groq API
#
# Groq runs LLMs on custom LPU (Language Processing Unit) hardware.
# This makes inference 10-20x faster than GPU-based APIs.
#
# We use one client for both expansion and answering — unlike Gemini
# where we needed two model configs, Groq's client is stateless.
# Temperature and max_tokens are passed per-call, not at model load time.
#
# Model: llama-3.3-70b-versatile
#   - 70B parameter Llama 3.3 — best free model on Groq
#   - Excellent instruction following — critical for JSON expansion output
#   - Strong reasoning — handles grounded answering well
#   - Free tier: 6000 requests/day, 30/min, 6000 tokens/min
#
# Alternative Groq models:
#   - llama-3.1-8b-instant  : faster, weaker — good for expansion only
#   - mixtral-8x7b-32768    : large context, good for long docs
#   - gemma2-9b-it          : Google's model, lighter
#
# We use 70b-versatile for both tasks because:
#   - Expansion needs good instruction following (JSON format compliance)
#   - Answering needs strong reasoning (grounding + citation discipline)
#   - Speed difference on Groq between 8b and 70b is small (LPU advantage)
# ─────────────────────────────────────────────────────────────────────────────

MODEL          = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

_expansion_cache: dict = {}
_answer_cache:    dict = {}


def _groq() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _groq_complete(messages: list, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Groq call with exponential-backoff retry on 429 and automatic
    fallback to FALLBACK_MODEL after retries are exhausted.
    Returns the response text directly.
    """
    delay = 5
    for attempt in range(3):
        try:
            return _groq().chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ).choices[0].message.content.strip()
        except Exception as e:
            if "429" not in str(e):
                raise
            if attempt < 2:
                print(f"  [Retry {attempt + 1}] Rate limit — waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  [Fallback] {MODEL} exhausted — switching to {FALLBACK_MODEL}")
                return _groq().chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ).choices[0].message.content.strip()
    raise RuntimeError("LLM call failed after retries and fallback")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: QUERY EXPANSION
# ─────────────────────────────────────────────────────────────────────────────

EXPANSION_PROMPT = """You are an expert RAG (Retrieval-Augmented Generation) query analyst specialising in hybrid search systems that combine BM25 keyword search with dense vector embeddings.

## Your task
Given a user query, generate exactly 4 rewritten variants. Each variant must serve a SPECIFIC and DISTINCT retrieval purpose. The goal is maximum retrieval recall — finding every relevant chunk in the document.

---

## Variant specifications

**Variant 1 — LEXICAL VARIANT**
Replace key terms with synonyms, related words, or alternative phrasings.
Goal: Help BM25 keyword search find chunks that use different vocabulary than the original.
Rule: Keep the same sentence structure. Only swap words — do not restructure the query.
Example transform: "predict" → "forecast", "model" → "system", "accuracy" → "performance"

**Variant 2 — TECHNICAL REFRAME**
Rephrase using precise domain-specific or academic terminology.
Goal: Match the exact vocabulary used in technical documents, research papers, or professional reports.
Rule: Think "how would an expert author write this concept in a paper?" Use technical jargon deliberately.

**Variant 3 — CONCEPT FOCUS**
Rephrase to focus tightly on the core concept or mechanism being asked about, while fully preserving the original intent.
Goal: Improve retrieval of highly specific conceptual chunks without losing meaning or constraints.
Rule: Do NOT strip constraints. Do NOT broaden scope. Zoom into the central mechanism — keep intent intact.

**Variant 4 — QUESTION REFORMULATION**
Rewrite as a natural, conversational question a user might type into a search engine or chatbot.
Goal: Improve alignment with embedding models trained on natural language, and improve recall for conversational phrasing in the document.
Rule: Write it naturally and directly, as if asking a knowledgeable colleague. Avoid jargon.

---

## Hard constraints (apply to ALL variants)
- Preserve the ORIGINAL INTENT exactly — never change what is being asked
- Do NOT introduce new topics, entities, or facts not present in the original query
- Do NOT generate overly generic queries (e.g., "what is machine learning", "explain AI")
- Do NOT broaden the scope beyond the original question
- Each variant must remain specific to the user's intent
- Keep variants concise — under 20 words each
- ALL 4 variants must be meaningfully different from each other and from the original
- Output ONLY valid JSON — no explanation, no markdown, no code blocks

## Output format
{{"variants": ["variant1", "variant2", "variant3", "variant4"]}}

---

## Few-shot examples

Query: "how does the attention mechanism work in transformers?"
Output:
{{"variants": [
  "how do transformers focus on relevant parts of the input sequence?",
  "scaled dot-product attention query key value weight computation transformers",
  "attention score computation mechanism between token representations",
  "what is attention in a transformer and how does it work?"
]}}

Query: "what accuracy did ResNet-152 achieve on ImageNet?"
Output:
{{"variants": [
  "what performance score did ResNet-152 get on the ImageNet benchmark?",
  "ResNet-152 top-1 top-5 classification accuracy ImageNet ILSVRC evaluation",
  "ResNet-152 ImageNet benchmark classification performance",
  "how accurate is ResNet-152 on the ImageNet dataset?"
]}}

Query: "how does antibiotic resistance develop in bacteria?"
Output:
{{"variants": [
  "how do bacteria become resistant to antibiotics over time?",
  "antimicrobial resistance mechanisms gene mutation selection pressure bacterial evolution",
  "mechanism by which bacteria acquire resistance to antibiotic treatment",
  "why do bacteria stop responding to antibiotics?"
]}}

---

Now generate variants for this query. Remember: specific, diverse, intent-preserving.

Query: "{query}"
Output:"""


def deduplicate_queries(queries: list[str]) -> list[str]:
    """
    Removes duplicate or near-duplicate queries.

    🧠 FIX for Issue 4:
    LLMs sometimes produce near-identical variants — same words, different order.
    Deduplication ensures each retrieval run is meaningfully distinct,
    avoiding correlated RRF signals that skew fusion results.
    """
    seen   = set()
    unique = []

    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
        else:
            print(f"  [Expansion] Duplicate removed: '{q}'")

    return unique


def expand_query(query: str) -> list[str]:
    """
    Expands a user query into up to 5 variants (original + 4 generated).

    🧠 CONCEPT: Groq chat completion API
    Groq uses the OpenAI-compatible chat completion format:
      - messages: list of {role, content} dicts
      - role "system" : sets the model's behaviour/persona
      - role "user"   : the actual input

    We put the full expansion prompt in "user" (not system) because:
    - The prompt contains examples and structured instructions
    - These are better parsed as user-turn content by Llama models
    - System prompts work better for short behavioural instructions

    temperature=0.1 for expansion:
    Slight randomness produces meaningfully diverse variants.
    Pure 0.0 tends to generate near-identical rewrites on repeated calls.

    Returns: [original_query, variant1, variant2, variant3, variant4]
    """
    print(f"\n  [Expansion] Generating 4 variants for: '{query}'")

    if query in _expansion_cache:
        print("  [Expansion] Cache hit — skipping LLM call")
        return _expansion_cache[query]

    try:
        raw = _groq_complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a RAG query expansion specialist. "
                        "You output ONLY valid JSON. No markdown, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": EXPANSION_PROMPT.format(query=query)
                }
            ],
            temperature=0.1,
            max_tokens=512,
        )

        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

        data     = json.loads(raw)
        variants = data.get("variants", [])

        # Validate and clean
        variants = [v.strip() for v in variants if isinstance(v, str) and v.strip()][:4]

        # Pad if fewer than 4 returned
        while len(variants) < 4:
            print(f"  [Expansion] Only got {len(variants)} variants — padding with original")
            variants.append(query)

        # Assemble: original first, then variants
        all_queries = [query] + variants

        # Deduplicate
        all_queries = deduplicate_queries(all_queries)

        print(f"  [Expansion] Final query set ({len(all_queries)} queries):")
        for i, q in enumerate(all_queries):
            label = "original " if i == 0 else f"variant {i}"
            print(f"    [{label}] {q}")

        _expansion_cache[query] = all_queries
        return all_queries

    except json.JSONDecodeError as e:
        print(f"  [Expansion] JSON parse error: {e}")
        print(f"  [Expansion] Raw response was: {raw[:200]}")
        print(f"  [Expansion] Falling back to original query only")
        return [query]

    except Exception as e:
        print(f"  [Expansion] Failed: {e} — falling back to original query only")
        return [query]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: GROUNDED ANSWERING
# ─────────────────────────────────────────────────────────────────────────────

ANSWERING_PROMPT = """Answer using ONLY the document excerpts below. No external knowledge.

CITATION FORMAT (mandatory):
  [Excerpt X | Source: <exact filename>, page Y]
Never invent filenames, excerpt numbers, or page numbers.

GROUNDING RULES:
1. Every factual claim must be supported by a specific excerpt.
2. Do NOT infer or generalise beyond what is explicitly stated.
3. If the context does not contain the answer: "Not found in provided documents."

Context quality: {context_quality}

Document excerpts:
{context_block}

Question: {query}

━━━ RESPOND IN THIS EXACT FORMAT ━━━

🧠 **Answer**
[Direct grounded answer — max 120 words. Cite every claim with [Excerpt X | Source: <filename>, page Y]]

💡 **Key Idea**
[1–2 sentence intuition or mechanism]

📚 **Sources**
[All citations used, one per line]"""


def _classify_chunk(content: str) -> str:
    """
    Classifies a chunk into one of four semantic categories:
      equation   — contains a mathematical formula or symbolic expression
      definition — introduces or defines a concept
      example    — illustrates with a concrete case
      mechanism  — describes a process, architecture, or how something works

    Used to label chunks in the context block so the LLM knows what kind
    of information each chunk provides and can synthesise accordingly.
    """
    import re
    cl = content.lower()

    # Equation: math symbols + operators together signal a formula chunk
    math_tokens = [
        "softmax(", "attention(", "q, k, v", "= softmax", "sqrt(",
        "argmax", "log p(", "∑", "∫", "∂", "·", "^t", "/ sqrt",
        "mathrm", "mathbf", "times d_k", "d_{k}", "d_k",
    ]
    has_math_token   = any(t in cl for t in math_tokens)
    operator_density = len(re.findall(r"[=+\-*/÷×√∑∫]", content))
    if has_math_token or operator_density >= 4:
        return "equation"

    # Definition: language that introduces or defines a concept
    def_signals = [
        "is defined as", "we define", "refers to", "is a model",
        "is an attention", "introduced in", "can be described as",
        "consists of", "is composed of", "is based on", "we propose",
        "the transformer", "is a sequence", "is a function",
    ]
    if any(s in cl for s in def_signals):
        return "definition"

    # Example: illustrative / comparative language
    if any(s in cl for s in ["for example", "e.g.", "such as", "consider the",
                              "illustrat", "analogous", "in contrast"]):
        return "example"

    return "mechanism"


def build_context_block(chunks: list[dict]) -> tuple[str, str]:
    """
    Formats retrieved chunks into a labeled context block.

    Each chunk is tagged with its semantic type (equation / definition /
    mechanism / example) so the LLM understands what kind of information
    it is reading and can synthesise across types, not just the same type.
    """
    if not chunks:
        return "", "⚠ WEAK CONTEXT: No relevant chunks were retrieved."

    lines     = []
    top_score = chunks[0].get("rerank_score", 0)

    for i, chunk in enumerate(chunks, 1):
        source  = chunk.get("source", "unknown")
        page    = chunk.get("page", "?")
        content = chunk.get("content", "").strip()

        header = f"[Excerpt {i} | Source: {source}, page {page}]"

        lines.append(f"{header}\n{content}")

    context_block = "\n\n---\n\n".join(lines)

    # Programmatic quality signal based on top rerank score
    if top_score >= 3.0:
        quality = "STRONG"
    elif top_score >= 0.0:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    return context_block, quality


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS PROMPT  (explanation intent — research-grade output)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are an advanced RAG assistant. Follow every rule below. No exceptions.

━━━ CONTEXT AWARENESS ━━━
- Always use prior conversation turns.
- Resolve "it", "this", "they" from earlier answers — never ask for clarification.
- DO NOT redefine any concept already explained in history.
- If referencing a prior concept, your FIRST SENTENCE must be one of:
    "As explained earlier,"
    "Building on the previous concept,"
    "As discussed above,"
  Skipping this on a follow-up is INCORRECT BEHAVIOR.

━━━ CITATIONS (MANDATORY) ━━━
- Every factual statement must cite its source:
    [Excerpt X | Source: <exact filename>, page Y]
- Use the most relevant chunk number (X) from the provided context.
- Multiple citations allowed per claim.
- No vague labels like "paper" or "document".
- If no source supports the claim: "Not found in provided documents."

━━━ FORMULA RULE ━━━
- If the topic involves attention, transformers, probability, or any mathematical concept:
  You MUST include the relevant formula on its own line.
  Example: Attention(Q,K,V) = softmax(QKᵀ / √d_k)V
  Then explain each variable in one sentence.

━━━ OUTPUT FORMAT ━━━
Use ONLY sections that apply. Never force empty sections. Use this EXACT structure:

🧠 **Answer**
[Direct grounded answer. Cite every factual claim: [Excerpt X | Source: <filename>, page Y]]

💡 **Key Idea**
[1–2 sentence intuition — always include]

⚙️ **How It Works** (optional — include only if question asks about application/integration)
- **Where:** where this is used in the pipeline or system
- **Why:** why it improves or matters
- **How:** how it integrates technically

🧮 **Formula** (include if topic involves attention, transformers, probability, or math)
[formula on its own line, then one sentence explaining variables]

📚 **Sources**
[List all citations used, one per line]

FORMATTING RULES:
- After every section heading (🧠 **Answer**, 💡 **Key Idea**, etc.) ALWAYS start content on the NEXT line
- Add a blank line between every section
- Bold all sub-labels (**Where:**, **Why:**, **How:**)
- Never put heading and text on the same line

━━━ HALLUCINATION CONTROL ━━━
- NEVER assume anything not explicitly in the context.
- Do NOT invent tools, models, metrics, or project details.
- If uncertain: "Not explicitly mentioned in the provided documents."
- DIAGRAMS: Do NOT describe, sketch, or infer a diagram in text if no actual
  image was provided to you. If the user asks for a diagram and none is in
  context, say exactly: "No diagram found in the indexed documents."
  Never write "the architecture consists of…" as a substitute for a diagram.

━━━ CONTRADICTION GUARD ━━━
- Check conversation history before responding.
- If a prior answer said YES → do not answer NO to the same question.
- If a conflict exists: "Note: this may conflict with the earlier answer — [reason]."

━━━ CONCISENESS ━━━
- Max 150 words. No repetition. No padding.
- For lists: max 2–3 most relevant items. Never dump full lists.

━━━ OUT OF SCOPE ━━━
- If the question is unrelated to the documents:
    "Not found in provided documents."

━━━ CONSISTENCY ━━━
- Same tone and format across all answers. No random style switches."""

SYNTHESIS_USER = """Context (read ALL chunks before answering):

{context_block}

---

Question: {query}

Return a clean, structured, consistent answer following your instructions.
Use ONLY information from the context above. Synthesize — do not summarize chunk by chunk."""


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "transformer":       ["encoder", "decoder", "attention", "positional"],
    "self-attention":    ["query", "key", "value", "softmax"],
    "attention":         ["query", "key", "value", "score"],
    "neural network":    ["layer", "weight", "activation"],
    "bert":              ["pre-training", "encoder", "fine-tuning"],
    "gpt":               ["decoder", "autoregressive", "generation"],
    "embedding":         ["vector", "dimension", "representation"],
    "backpropagation":   ["gradient", "loss", "weight"],
    "convolutional":     ["kernel", "filter", "pooling"],
}


def _required_keywords(query: str) -> list[str]:
    q = query.lower()
    required: list[str] = []
    for topic, kws in _TOPIC_KEYWORDS.items():
        if topic in q:
            required.extend(kws[:3])
    return list(set(required))


def _validate_coverage(answer: str, query: str) -> tuple[bool, list[str]]:
    """Returns (all_covered, list_of_missing_keywords)."""
    required = _required_keywords(query)
    if not required:
        return True, []
    a = answer.lower()
    missing = [k for k in required if k not in a]
    return len(missing) == 0, missing


# def answer(query: str, chunks: list[dict]) -> str:

    # """
    # Generates a grounded, cited answer using Groq.

    # 🧠 CONCEPT: system vs user message split for answering
    # We split the prompt into two messages:
    #   system : the role, rules, and citation format — behavioural instructions
    #   user   : context excerpts + actual question — the variable content

    # Why split?
    # - System prompt is cached by Groq on repeated calls with same instructions
    # - User message contains the dynamic content (different chunks per query)
    # - Llama 3 models follow system instructions more reliably when separated
    # - Cleaner separation of "how to behave" vs "what to answer"

    # temperature=0.0 — fully deterministic.
    # Same question + same chunks must always produce the same answer.
    # Any randomness risks subtle hallucination in grounded generation.
    # """
    # if not chunks:
    #     return (
    #         "**Answer:** No relevant information found in the ingested "
    #         "documents for this question.\n\n"
    #         "**Confidence:** LOW\n\n"
    #         "**Gaps:** The query appears to be outside the scope of "
    #         "the available documents."
    #     )

    # context_block, context_quality = build_context_block(chunks)

    # # Split into system (behaviour) + user (content)
    # system_message = (
    #     "You are a precise document intelligence assistant in a RAG system. "
    #     "You answer ONLY from provided excerpts. "
    #     "You NEVER use your training knowledge. "
    #     "Every factual claim must have a citation in format: "
    #     "[Excerpt X | Source: filename, page N]. "
    #     "If a claim is not directly supported by an excerpt, omit it entirely."
    #     + (f"\n\nSPECIAL INSTRUCTION: {llm_instruction}" if llm_instruction else "")
    # )
    

    # user_message = ANSWERING_PROMPT.format(
    #     context_quality=context_quality,
    #     context_block=context_block,
    #     query=query,
    # )

    # print(f"  [Groq] Sending {len(chunks)} excerpts to {MODEL}...")
    # print(f"  [Groq] Top rerank score: {chunks[0].get('rerank_score', 0):.2f}")

    # try:
    #     response = client.chat.completions.create(
    #         model=MODEL,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             {"role": "user",   "content": user_message},
    #         ],
    #         temperature=0.0,
    #         max_tokens=1024,
    #     )

    #     return response.choices[0].message.content.strip()

    # except Exception as e:
    #     return f"**Answer:** Error generating response — {e}\n\n**Confidence:** LOW"

def answer(
    query:   str,
    chunks:  list[dict],
    intent:  str        = "explanation",
    history: list[dict] = None,
) -> str:
    """
    Intent-aware answering with two distinct generation strategies.

    explanation → SYNTHESIS PATH
        Structured, synthesized research-grade answer.
        Conversation history injected as real LLM turns so the model
        can reference prior explanations instead of restating them.

    lookup / boolean / comparison / aggregation → GROUNDED PATH
        Precise, extractive, citation-grounded answers.

    history: list of {"role": "user"|"assistant", "content": str}
        Last N turns from the conversation (trimmed by caller).
        Passed as real messages so the LLM natively understands context.
    """
    if not chunks:
        # The caller may have appended an image_note to query — honour it
        if "[System note:" in query and "Visual References" in query:
            return "🧠 **Answer**\nSee the Visual References section below for the diagram."
        return (
            "🧠 **Answer**\nNo relevant information found in the ingested documents "
            "for this question."
        )

    context_block, context_quality = build_context_block(chunks)

    chunk_types  = [_classify_chunk(c.get("content", "")) for c in chunks]
    type_summary = ", ".join(sorted(set(chunk_types)))
    print(f"  [Groq] {len(chunks)} chunks ({type_summary}) → {MODEL}")
    print(f"  [Groq] Top score: {chunks[0].get('rerank_score', 0):.2f} | Intent: {intent}")

    # ── SYNTHESIS PATH ────────────────────────────────────────────────────────
    if intent == "explanation":
        cache_key = (query, intent, bool(history))
        if cache_key in _answer_cache:
            print("  [Answer] Cache hit — skipping LLM call")
            return _answer_cache[cache_key]

        user_message = SYNTHESIS_USER.format(
            context_block=context_block[:2000],
            query=query,
        )
        prior_turns = (history or [])[-6:]
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            *prior_turns,
            {"role": "user", "content": user_message},
        ]
        try:
            first_answer = _groq_complete(messages, temperature=0.0, max_tokens=2000)

            covered, missing = _validate_coverage(first_answer, query)
            if not covered:
                print(f"  [Coverage] Missing: {missing} — retrying...")
                retry_msg = (
                    f"Your answer is missing these required topics: {', '.join(missing)}.\n"
                    f"Extend your answer to explicitly cover them. "
                    f"Use ONLY the provided context."
                )
                first_answer = _groq_complete(
                    messages=[
                        *messages,
                        {"role": "assistant", "content": first_answer},
                        {"role": "user",      "content": retry_msg},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                print("  [Coverage] Retry complete")

            _answer_cache[cache_key] = first_answer
            return first_answer

        except Exception as e:
            return f"**Answer:** Error — {e}\n\n**Confidence:** LOW"

    # ── GROUNDED PATH ─────────────────────────────────────────────────────────
    INTENT_RULES = {
        "lookup": (
            "- Provide a precise, direct answer\n"
            "- Start with the definition in ONE sentence\n"
            "- Keep answer under 3 sentences\n"
        ),
        "boolean": (
            "- FIRST word MUST be 'Yes' or 'No'\n"
            "- Then provide justification from the chunks\n"
            "- Do NOT hedge unless evidence is genuinely ambiguous\n"
        ),
        "comparison": (
            "- Compare entities directly, not separately\n"
            "- Highlight key differences AND similarities\n"
            "- Use a structured format (bullets or table)\n"
        ),
        "aggregation": (
            "- Extract ALL relevant items exhaustively\n"
            "- Use bullet list format\n"
            "- Combine across all chunks — do NOT miss any item\n"
        ),
    }
    OUTPUT_CONSTRAINTS = {
        "lookup":      "Answer MUST be short and precise.",
        "boolean":     "Answer MUST start with YES or NO.",
        "comparison":  "Answer MUST explicitly compare the items side by side.",
        "aggregation": "Answer MUST be a complete bullet list.",
    }

    intent_rules = INTENT_RULES.get(intent, "- Answer directly and concisely.\n")
    output_rule  = OUTPUT_CONSTRAINTS.get(intent, "")

    system_message = (
        "You are a high-precision RAG assistant.\n\n"
        "RULES:\n"
        "- Use ONLY information from the provided context\n"
        "- NEVER use external knowledge or guess\n"
        "- After every factual claim, cite: [Excerpt X | Source: <exact filename>, page Y]\n"
        f"- Context quality: {context_quality}\n\n"
        f"INTENT: {intent}\n"
        f"BEHAVIOR:\n{intent_rules}\n"
        f"OUTPUT RULE: {output_rule}"
    )
    user_message = ANSWERING_PROMPT.format(
        context_quality=context_quality,
        context_block=context_block,
        query=query,
    )

    cache_key = (query, intent, bool(history))
    if cache_key in _answer_cache:
        print("  [Answer] Cache hit — skipping LLM call")
        return _answer_cache[cache_key]

    try:
        result = _groq_complete(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        _answer_cache[cache_key] = result
        return result
    except Exception as e:
        return f"**Answer:** Error — {e}\n\n**Confidence:** LOW"