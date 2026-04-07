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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


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

    try:
        response = client.chat.completions.create(
            model=MODEL,
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

        # ── Extract and parse response ────────────────────────────────────
        # 🧠 CONCEPT: Groq response structure
        # response.choices[0].message.content → the generated text string
        # Same structure as OpenAI — Groq is OpenAI API-compatible.
        # This means switching from OpenAI to Groq is literally one line:
        # change the client init. Everything else stays identical.
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them despite instructions
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

ANSWERING_PROMPT = """You are a precise document intelligence assistant operating in a RAG (Retrieval-Augmented Generation) system.

## Your role
Answer the user's question using ONLY the numbered document excerpts provided below.
You have access to NO other information source. Your training knowledge does not exist for this task.

## Strict grounding rules
1. ONLY use information explicitly stated in the excerpts — nothing else
2. If a claim cannot be directly supported by a specific excerpt, do NOT include it
3. Do NOT generalise beyond what is explicitly stated, even if it seems logically obvious
4. Do NOT infer, assume, or extrapolate — only report what the text directly says
5. If the excerpts do not contain the answer, say so clearly and specifically

## Citation format
After EVERY factual claim, cite the source using this exact format:
[Excerpt X | Source: filename, page N]

Where X is the excerpt number shown in the context below.
Use this format consistently — do not invent page numbers or excerpt numbers.

## Confidence levels
- HIGH   = excerpt directly and completely answers the question with explicit statements
- MEDIUM = excerpt partially answers or implies the answer — state what is missing
- LOW    = excerpts are tangentially related — answer is uncertain, flag this clearly

## Context quality
{context_quality}

## Document excerpts (retrieved and ranked by relevance to your question)
{context_block}

## User question
{query}

## Response format

**Answer:**
[Your grounded answer. Every factual claim followed by [Excerpt X | Source: filename, page N].
If the context does not support a complete answer, state exactly what is and is not covered.]

**Key points:**
- [specific point] [Excerpt X | Source: filename, page N]
- [specific point] [Excerpt X | Source: filename, page N]

**Confidence:** [HIGH / MEDIUM / LOW]

**Gaps:**
[Specific information that would improve the answer but is not present in the excerpts.
Write "None" if the excerpts fully answer the question.]"""


def build_context_block(chunks: list[dict]) -> tuple[str, str]:
    """
    Builds the numbered context block passed to the answering LLM.

    Numbers each excerpt so the model can cite precisely:
    [Excerpt 1 | Source: paper.pdf, page 2]
    instead of hallucinating page numbers from memory.
    """
    if not chunks:
        return "", "⚠ WEAK CONTEXT: No relevant chunks were retrieved."

    lines     = []
    top_score = chunks[0].get("rerank_score", 0)

    for i, chunk in enumerate(chunks, 1):
        score   = round(chunk.get("rerank_score", 0), 2)
        source  = chunk.get("source", "unknown")
        page    = chunk.get("page", "?")
        content = chunk.get("content", "").strip()

        lines.append(
            f"[Excerpt {i}] Source: {source}, Page {page} | "
            f"Relevance score: {score}\n"
            f"{content}"
        )

    context_block = "\n\n---\n\n".join(lines)

    # Quality signal calibrates model confidence
    if top_score >= 3.0:
        quality = "✅ STRONG: Top excerpts are highly relevant. Answer with HIGH confidence where supported."
    elif top_score >= 0.0:
        quality = "⚠ MODERATE: Excerpts partially relevant. Use MEDIUM confidence and note gaps explicitly."
    else:
        quality = "❌ WEAK: Low relevance scores. Use LOW confidence. Prioritise transparency over completeness."

    return context_block, quality


def answer(query: str, chunks: list[dict]) -> str:
    """
    Generates a grounded, cited answer using Groq.

    🧠 CONCEPT: system vs user message split for answering
    We split the prompt into two messages:
      system : the role, rules, and citation format — behavioural instructions
      user   : context excerpts + actual question — the variable content

    Why split?
    - System prompt is cached by Groq on repeated calls with same instructions
    - User message contains the dynamic content (different chunks per query)
    - Llama 3 models follow system instructions more reliably when separated
    - Cleaner separation of "how to behave" vs "what to answer"

    temperature=0.0 — fully deterministic.
    Same question + same chunks must always produce the same answer.
    Any randomness risks subtle hallucination in grounded generation.
    """
    if not chunks:
        return (
            "**Answer:** No relevant information found in the ingested "
            "documents for this question.\n\n"
            "**Confidence:** LOW\n\n"
            "**Gaps:** The query appears to be outside the scope of "
            "the available documents."
        )

    context_block, context_quality = build_context_block(chunks)

    # Split into system (behaviour) + user (content)
    system_message = (
        "You are a precise document intelligence assistant in a RAG system. "
        "You answer ONLY from provided excerpts. "
        "You NEVER use your training knowledge. "
        "Every factual claim must have a citation in format: "
        "[Excerpt X | Source: filename, page N]. "
        "If a claim is not directly supported by an excerpt, omit it entirely."
        + (f"\n\nSPECIAL INSTRUCTION: {llm_instruction}" if llm_instruction else "")
    )
    

    user_message = ANSWERING_PROMPT.format(
        context_quality=context_quality,
        context_block=context_block,
        query=query,
    )

    print(f"  [Groq] Sending {len(chunks)} excerpts to {MODEL}...")
    print(f"  [Groq] Top rerank score: {chunks[0].get('rerank_score', 0):.2f}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"**Answer:** Error generating response — {e}\n\n**Confidence:** LOW"