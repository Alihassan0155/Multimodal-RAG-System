# # src/pipeline/chunker.py

# import json
# import tiktoken
# from pathlib import Path
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# # ─────────────────────────────────────────────────────────────────────────────
# # 🧠 CONCEPT: tiktoken encoder
# #
# # tiktoken converts text → list of integer token IDs.
# # "cl100k_base" is the encoding used by text-embedding-3 and GPT-4.
# # We use it here purely for COUNTING tokens, not for encoding.
# #
# # Why count tokens instead of characters?
# # "GPT-4"    = 3 chars, 1 token  (treated as one unit)
# # "🔥"       = 4 bytes, 1 token
# # "Pneumonia" = 9 chars, 3 tokens (split into sub-words)
# # Character-based splitting gives inconsistent chunk sizes.
# # Token-based splitting matches what the embedding model actually sees.
# #
# # Alternatives:
# # - spacy tokenizer (linguistic, language-aware)
# # - huggingface AutoTokenizer (matches your specific model exactly)
# # - character count / 4 (rough heuristic, inaccurate)
# # We use tiktoken because it's fast, accurate for OpenAI-family models,
# # and requires zero configuration.
# # ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER = tiktoken.get_encoding("cl100k_base")


# def count_tokens(text: str) -> int:
#     """Count how many tokens a string contains."""
#     return len(TOKENIZER.encode(text))


# def build_splitter(chunk_size: int = 384, chunk_overlap: int = 40):
#     """
#     Creates a RecursiveCharacterTextSplitter configured for our pipeline.

#     🧠 CONCEPT: RecursiveCharacterTextSplitter
#     Tries to split on separators in this order:
#        1. "\n\n"  → paragraph breaks (preferred — most natural)
#        2. "\n"    → line breaks
#        3. ". "    → sentence boundaries
#        4. " "     → word boundaries
#        5. ""      → characters (last resort, almost never needed)

#     It only moves to the next separator if the current chunk is still
#     too large after splitting on the current one.

#     WHY THIS ORDER:
#     Paragraph → sentence → word preserves semantic coherence.
#     A chunk about "ResNet accuracy" won't be contaminated with half
#     a sentence about "EfficientNet training" from the next paragraph.

#     chunk_size=512     : target size in tokens
#     chunk_overlap=50   : tokens shared between adjacent chunks
#     length_function    : we pass count_tokens so it measures tokens,
#                          not characters (this is critical)
#     is_separator_regex : False means treat separators as literal strings,
#                          not regex patterns
#     """
#     return RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=count_tokens,
#         separators=["\n\n", "\n", ". ", " ", ""],
#         is_separator_regex=False,
#     )


# def chunk_page(
#     page: dict,
#     splitter: RecursiveCharacterTextSplitter,
# ) -> list[dict]:
#     """
#     Takes one page dict (from extract_text.py) and returns a list of chunks.

#     Input page dict shape:
#     {
#         "doc_id": "a3f8c1d2",
#         "source": "paper.pdf",
#         "page": 3,
#         "text": "Convolutional neural networks...",
#         "has_tables": True,
#         "table_regions": [...],
#         ...
#     }

#     Output chunk shape:
#     {
#         "id":          "a3f8c1d2_text_p3_0",
#         "doc_id":      "a3f8c1d2",
#         "source":      "paper.pdf",
#         "page":        3,
#         "type":        "text",
#         "chunk_index": 0,
#         "content":     "Convolutional neural networks...",
#         "token_count": 310,
#         "has_tables":  True,   ← carried forward for Phase 2 awareness
#     }
#     """
#     text = page["text"]

#     # Edge case: page text is under the chunk size — no splitting needed
#     # We still wrap it in our chunk schema for consistency downstream
#     if count_tokens(text) <= 512:
#         return [_build_chunk(page, text, chunk_index=0)]

#     # Split the page text into multiple chunks
#     splits = splitter.split_text(text)

#     chunks = []
#     for i, split_text in enumerate(splits):

#         # Skip empty or near-empty splits
#         # These can occur when pages have lots of whitespace or formatting
#         if count_tokens(split_text.strip()) < 20:
#             continue

#         chunks.append(_build_chunk(page, split_text.strip(), chunk_index=i))

#     return chunks


# def _build_chunk(page: dict, content: str, chunk_index: int) -> dict:
#     """
#     Assembles the final chunk dict.

#     WHY A SEPARATE HELPER:
#     Both the "fits in one chunk" path and the "split into many" path
#     need to produce the same shaped dict. Centralising it here means
#     we can never accidentally diverge between the two paths.

#     ID FORMAT: "{doc_id}_text_p{page}_{index}"
#     Example:   "a3f8c1d2_text_p3_0"

#     Why encode type and page in the ID?
#     → Debugging. When you see "a3f8c1d2_table_p7_2" in a retrieval result
#       you immediately know: document a3f8c1d2, a table, from page 7,
#       third chunk on that page. No lookup needed.
#     """
#     doc_id = page["doc_id"]
#     page_num = page["page"]

#     return {
#         "id":          f"{doc_id}_text_p{page_num}_{chunk_index}",
#         "doc_id":      doc_id,
#         "source":      page["source"],
#         "page":        page_num,
#         "type":        "text",
#         "chunk_index": chunk_index,
#         "content":     content,
#         "token_count": count_tokens(content),
#         # Carry forward table flag — useful when Phase 2 chunks
#         # need to know their neighbouring page had tables
#         "has_tables":  page.get("has_tables", False),
#     }


# def chunk_document(pages: list[dict], chunk_size: int = 384, chunk_overlap: int = 40) -> list[dict]:
#     """
#     Processes all pages from one document and returns all chunks.

#     This is the main function you call from outside this module.

#     WHY PROCESS ALL PAGES TOGETHER:
#     Future phases (tables, images) will also produce chunks.
#     Having one function that handles a whole document makes it easy
#     to merge text chunks + table chunks + image chunks into one list
#     before sending to the vector DB.
#     """
#     splitter = build_splitter(chunk_size, chunk_overlap)
#     all_chunks = []

#     for page in pages:
#         page_chunks = chunk_page(page, splitter)
#         all_chunks.extend(page_chunks)

#     return all_chunks


# def save_chunks(chunks: list[dict], output_dir: str = "data/extracted/text"):
#     """
#     Saves chunks to JSON alongside the extracted pages file.

#     WHY SAVE CHUNKS SEPARATELY FROM PAGES:
#     Extraction (pages) and chunking are two distinct pipeline stages.
#     Saving both separately means:
#     - You can re-chunk with different settings without re-extracting
#     - You can inspect chunk quality without re-running the whole pipeline
#     - Debugging one stage doesn't require touching the other
#     """
#     if not chunks:
#         print("⚠ No chunks to save.")
#         return

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     doc_id = chunks[0]["doc_id"]
#     source = chunks[0]["source"].replace(".pdf", "")
#     output_path = output_dir / f"{source}_{doc_id}_chunks.json"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2, ensure_ascii=False)

#     print(f"✓ Saved {len(chunks)} chunks → {output_path}")
#     return str(output_path)


# # ── Quick test runner ─────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python -m src.pipeline.chunker <path_to_extracted_json>")
#         print("Example: python -m src.pipeline.chunker data/extracted/text/paper_a3f8c1d2.json")
#         sys.exit(1)

#     # Load the extracted pages JSON produced by extract_text.py
#     with open(sys.argv[1], "r", encoding="utf-8") as f:
#         pages = json.load(f)

#     print(f"📄 Loaded {len(pages)} pages from {sys.argv[1]}")

#     # Chunk
#     chunks = chunk_document(pages)

#     # Stats
#     token_counts = [c["token_count"] for c in chunks]
#     print(f"\n✓ Created {len(chunks)} chunks")
#     print(f"  Min tokens : {min(token_counts)}")
#     print(f"  Max tokens : {max(token_counts)}")
#     print(f"  Avg tokens : {sum(token_counts) // len(token_counts)}")

#     # Preview first 3 chunks
#     print("\n── First 3 chunks preview ───────────────────────────────────")
#     for chunk in chunks[:3]:
#         print(f"\n  id          : {chunk['id']}")
#         print(f"  page        : {chunk['page']}")
#         print(f"  token_count : {chunk['token_count']}")
#         print(f"  content     : {chunk['content'][:120]}...")

#     # Save
#     save_chunks(chunks)

# src/pipeline/chunker.py

import re
import tiktoken


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Pattern-based heading detection — no hardcoding
#
# Instead of maintaining a keyword list, we score every line based on
# STRUCTURAL properties. A heading looks a certain way regardless of
# what domain the document is in.
#
# Signals used (all case-insensitive via .lower() where needed):
#   - Numbered prefix     : "1.", "2.", "1.1", "Step 1:", "Phase 2"
#   - Line length         : headings are short relative to body text
#   - Ending punctuation  : headings rarely end with "." or ","
#   - Bullet markers      : headings are never bullets
#   - Capitalisation      : headings often start with uppercase
#   - Colon at end        : "Input:", "Goal:", "Overview:"
#   - Context (next line) : a blank line after short text = heading
#
# THRESHOLD = 50 out of 100.
# Tune this down (e.g. 40) if too few headings detected.
# Tune this up (e.g. 60) if too many false headings detected.
# ─────────────────────────────────────────────────────────────────────────────

TOKENIZER      = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS     = 512
MIN_TOKENS     = 30
HEAD_THRESHOLD = 50  # minimum score to be classified as a heading

# Numbered heading pattern — no keywords, pure structure
# Matches: "1.", "2.", "1.1.", "1.1.2", "Step 1:", "Phase 2 —", "(1)"
NUMBERED = re.compile(
    r"^(\(?\d+[\.\)]\s|\d+\.\d+\.?\s|step\s+\d+\s*[:\-–]?|phase\s+\d+\s*[:\-–]?)",
    re.IGNORECASE,
)

# All-caps line (at least 3 chars, no digits dominating)
ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\(\)\-\:]{2,}$")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def score_heading(line: str, next_line: str = "") -> int:
    """
    Scores a line 0–100 based on structural heading signals.
    No keywords. No hardcoding. Works on any document domain.

    Args:
        line      : the line being evaluated
        next_line : the line immediately after (context signal)

    Returns: integer score 0–100
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 2:
        return 0

    score = 0
    lower = stripped.lower()  # lowercase for case-insensitive pattern checks

    # ── Signal 1: Numbered prefix (+40) ──────────────────────────────────
    # Strongest signal — "1.", "2.1", "Step 3:" are almost always headings
    if NUMBERED.match(lower):
        score += 40

    # ── Signal 2: Short line (+20) ────────────────────────────────────────
    # Headings are concise. Body text runs long.
    # Threshold 65 chars ~ one line of text at normal font size
    if len(stripped) < 65:
        score += 20
    elif len(stripped) < 90:
        score += 8  # partial credit for medium-length lines

    # ── Signal 3: No ending punctuation (+15) ────────────────────────────
    # Body sentences end with ".", ",", ";", "!", "?"
    # Headings almost never do — they stand alone
    if not stripped.endswith((".", ",", ";", "!", "?")):
        score += 15

    # ── Signal 4: Not a bullet (+10) ─────────────────────────────────────
    # Bullets are content, never headings
    if not stripped.startswith(("●", "•", "-", "*", "○", "–", "◦")):
        score += 10

    # ── Signal 5: Starts with uppercase (+10) ────────────────────────────
    # Headings almost always start with a capital letter
    if stripped[0].isupper():
        score += 10

    # ── Signal 6: Ends with colon (+15) ──────────────────────────────────
    # "Input:", "Goal:", "Overview:" — strong heading pattern
    if stripped.endswith(":"):
        score += 15

    # ── Signal 7: ALL CAPS line (+15) ────────────────────────────────────
    # "INTRODUCTION", "METHODS" — traditional academic heading style
    if ALL_CAPS.match(stripped):
        score += 15

    # ── Signal 8: Followed by blank line (+10) ────────────────────────────
    # A short line followed by a blank line is almost certainly a heading
    # that separates two blocks of content
    if not next_line.strip() and len(stripped) < 65:
        score += 10

    # ── Signal 9: Contains only a few words (+5) ─────────────────────────
    # Headings are typically 1–6 words. Long lines are body text.
    word_count = len(stripped.split())
    if 1 <= word_count <= 6:
        score += 5

    # ── Penalty: looks like a data row (–20) ─────────────────────────────
    # Lines with multiple numbers are likely table rows or data, not headings
    # e.g. "ResNet50 | 76.1% | 25M"
    if len(re.findall(r"\d+", stripped)) >= 3:
        score -= 20

    # ── Penalty: very long line (–15) ────────────────────────────────────
    if len(stripped) > 120:
        score -= 15

    return max(score, 0)  # floor at 0


# ── Stage 1: Context-aware section splitting ──────────────────────────────────

def split_into_sections(text: str) -> list[dict]:
    """
    Splits page text into sections using pattern-based heading detection.

    🧠 How it works:
    1. Split text into lines
    2. Score every line using score_heading()
    3. Lines scoring >= HEAD_THRESHOLD open a new section
    4. Bullet clusters (3+ consecutive bullets) are kept together —
       splitting mid-list destroys context
    5. Double blank lines act as soft paragraph boundaries within a section

    Returns:
    [
        {
            "heading": "1. Input: Bacterial Genomic Data",
            "content": "The system takes bacterial genome...",
            "type":    "section" | "bullets" | "paragraph"
        },
        ...
    ]
    """
    lines    = text.split("\n")
    sections = []

    current_heading = ""
    current_lines   = []
    current_type    = "paragraph"
    bullet_streak   = 0

    def flush():
        """Save current buffer as a completed section."""
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({
                "heading": current_heading,
                "content": content,
                "type":    current_type,
            })

    for i, line in enumerate(lines):
        stripped  = line.strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # ── Blank line ────────────────────────────────────────────────────
        if not stripped:
            current_lines.append("")
            bullet_streak = 0
            continue

        # ── Bullet detection ──────────────────────────────────────────────
        is_bullet = stripped.startswith(("●", "•", "-", "*", "○", "–", "◦"))
        if is_bullet:
            bullet_streak += 1
            # Once we are in a bullet cluster keep accumulating
            if current_type != "bullets" and bullet_streak >= 2:
                flush()
                current_heading = current_heading
                current_lines   = [stripped]
                current_type    = "bullets"
                continue
        else:
            bullet_streak = 0

        # ── Heading detection (pattern scored, case-insensitive) ──────────
        heading_score = score_heading(stripped, next_line)

        if heading_score >= HEAD_THRESHOLD and not is_bullet:
            flush()
            # Reset buffer with this heading
            current_heading = stripped
            current_lines   = []
            current_type    = "section"
            continue

        current_lines.append(stripped)

    flush()  # final buffer
    return sections


# ── Stage 2: Recursive fallback for oversized sections ───────────────────────

def recursive_split(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Recursively splits text that exceeds max_tokens.

    🧠 CONCEPT: separator priority
    Tries to split at the most natural boundary first:
      "\n\n" paragraph  →  "\n" line  →  ". " sentence  →  " " word

    Only moves to a finer split if the chunk is still too big.
    This preserves as much semantic coherence as possible.

    Used ONLY as a fallback when a section is genuinely oversized.
    The vast majority of sections from stage 1 won't need this.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    for separator in ["\n\n", "\n", ". ", " "]:
        parts = text.split(separator)
        if len(parts) <= 1:
            continue

        chunks  = []
        current = ""

        for part in parts:
            candidate = (current + separator + part).strip() if current else part.strip()
            if count_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part.strip()

        if current:
            chunks.append(current)

        if len(chunks) > 1:
            return chunks

    # Absolute last resort: split in half
    mid = len(text) // 2
    return [text[:mid].strip(), text[mid:].strip()]


# ── Section merging ───────────────────────────────────────────────────────────

def merge_small_sections(sections: list[dict], min_tokens: int = MIN_TOKENS) -> list[dict]:
    """
    Merges sections too small to be meaningful on their own.

    🧠 WHY:
    A 10-token section like "Goal:" with no body text is useless to the
    retriever. Merging it forward into the next section preserves the
    heading as context while ensuring the chunk has real substance.
    """
    merged  = []
    pending = None

    for section in sections:
        tokens = count_tokens(section["content"])

        if pending is not None:
            # Prepend pending into current section
            section["content"] = (
                pending["heading"] + "\n" + pending["content"]
                + "\n\n" + section["content"]
            ).strip()
            if not section["heading"]:
                section["heading"] = pending["heading"]
            pending = None

        if tokens < min_tokens:
            pending = section
        else:
            merged.append(section)

    # Flush any trailing small section
    if pending is not None:
        if merged:
            merged[-1]["content"] += "\n\n" + pending["content"]
        else:
            merged.append(pending)

    return merged


# ── Chunk assembly ────────────────────────────────────────────────────────────

def build_chunk(
    section:     dict,
    content:     str,
    page:        dict,
    chunk_index: int,
    total:       int,
) -> dict:
    """
    Assembles the final chunk dict.

    🧠 Title-enhanced embedding:
    The heading is prepended to the content before embedding.
    This means a query like "what is the input to the system?" will
    match a chunk whose heading is "1. Input: Bacterial Genomic Data"
    even if the body text doesn't repeat the word "input".

    ID: {doc_id}_section_p{page}_{index}
    """
    doc_id   = page["doc_id"]
    page_num = page["page"]

    # full_content = (
    #     f"{section['heading']}\n{content}".strip()
    #     if section["heading"]
    #     else content.strip()
    # )
    content_clean = content.strip()
    heading       = section["heading"].strip()

    if heading and content_clean.lower().startswith(heading.lower()):
    # Heading already present in content — don't prepend again
        full_content = content_clean
    else:
    # Heading not in content — prepend it for title-enhanced embedding
        full_content = f"{heading}\n{content_clean}".strip() if heading else content_clean
    
    return {
        "id":           f"{doc_id}_section_p{page_num}_{chunk_index}",
        "doc_id":       doc_id,
        "source":       page["source"],
        "page":         page_num,
        "type":         section["type"],
        "heading":      section["heading"],
        "chunk_index":  chunk_index,
        "total_chunks": total,
        "content":      full_content,
        "token_count":  count_tokens(full_content),
        "has_tables":   page.get("has_tables", False),
    }


# ── Main public function ──────────────────────────────────────────────────────

def chunk_document(pages: list[dict]) -> list[dict]:
    """
    Full hybrid chunking pipeline.

    For each page:
      1. Pattern-based section detection  (no keywords, no hardcoding)
      2. Merge tiny sections
      3. Recursive split on oversized sections
      4. Assemble chunk dicts with full metadata
    """
    all_chunks = []

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        print(f"\n  Page {page['page']}:")

        # Stage 1
        sections = split_into_sections(text)
        print(f"    Sections detected  : {len(sections)}")

        # Merge small
        sections = merge_small_sections(sections)
        print(f"    After merging small: {len(sections)}")

        # Stage 2 + build
        page_chunks = []
        for section in sections:
            content = section["content"]
            if count_tokens(content) > MAX_TOKENS:
                sub_contents = recursive_split(content)
                print(f"    Recursive split '{section['heading'][:25]}' "
                      f"→ {len(sub_contents)} sub-chunks")
            else:
                sub_contents = [content]

            for sub in sub_contents:
                if sub.strip():
                    page_chunks.append((section, sub))

        total = len(page_chunks)
        for i, (section, content) in enumerate(page_chunks):
            all_chunks.append(build_chunk(section, content, page, i, total))

        print(f"    Final chunks       : {total}")

    return all_chunks