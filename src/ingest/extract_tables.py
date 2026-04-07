# src/ingest/extract_tables.py
import re
import io
import os
import base64
import json
from pathlib import Path
import warnings
import camelot
import fitz  # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

# Suppress Camelot temp file cleanup errors on Windows
warnings.filterwarnings("ignore", category=ResourceWarning)

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Camelot extraction modes
#
# Camelot has two algorithms:
#
# LATTICE mode:
#   Detects tables by finding horizontal and vertical LINES in the PDF.
#   Works on tables with visible borders/grid lines.
#   Most reliable — structured output, correct cell mapping.
#   Fails on: borderless tables, image tables.
#
# STREAM mode:
#   Detects tables by analysing WHITESPACE between text columns.
#   Works on borderless tables where columns are aligned by spacing.
#   Less reliable — whitespace heuristics can misalign cells.
#   Falls back to this when lattice finds nothing.
#
# We try lattice first, then stream as fallback.
# This covers the majority of real-world PDF tables.
#
# Alternative libraries:
#   tabula-py : Java-based, similar to Camelot, less accurate on complex tables
#   pdfplumber: we already use it — its table extraction is weaker than Camelot
#   PDFTables : cloud API, paid
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Groq Vision for image tables
#
# Some PDFs embed tables as raster images (PNG/JPEG) — no text layer at all.
# pdfplumber returns nothing. Camelot returns nothing.
# The only solution: extract the image → send to a vision-capable LLM.
#
# Groq currently supports vision via llama-3.2-11b-vision-preview.
# We send the image as base64 in the message content.
#
# WHY BASE64:
# APIs can't receive raw binary files in JSON. Base64 encodes binary
# data as ASCII text — safe to embed in any JSON payload.
# The receiving API decodes it back to the image before processing.
#
# Alternative vision models:
#   GPT-4V         : best quality, paid
#   Claude 3.5     : excellent, paid  
#   Gemini 1.5     : free tier, good quality
#   LLaVA (local)  : free, runs on CPU, weaker quality
# We use Groq because it's already wired up and free.
# ─────────────────────────────────────────────────────────────────────────────

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ── Prompt for vision-based table extraction ──────────────────────────────────

TABLE_VISION_PROMPT = """You are a precise document data extraction specialist.
The image shows a table from a document. Extract ALL information from this table.

## Your task
1. Identify the table structure (headers, rows, columns)
2. Generate a natural language summary describing what the table covers
3. Extract the raw data in CSV format

## Rules
- Extract EVERY row and column — do not skip any data
- Preserve exact text from the table — do not paraphrase cell values
- If a cell is empty, use an empty string in CSV
- If headers span multiple rows, combine them sensibly

## Output format (JSON only, no markdown)
{
  "summary": "Natural language description of what this table contains and its key information",
  "headers": ["col1", "col2", "col3"],
  "rows": [["val1", "val2", "val3"], ["val1", "val2", "val3"]],
  "csv": "col1,col2,col3\\nval1,val2,val3\\n..."
}"""


TABLE_SUMMARY_PROMPT = """You are a document intelligence assistant.
Below is a table extracted from a PDF document (page {page}).

## Table data (CSV format)
{csv_data}

## Your task
Write a natural language summary of this table that:
1. Describes what the table is about in one sentence
2. Highlights the most important values, patterns, or relationships
3. Mentions key terms, numbers, or findings explicitly
4. Is specific enough that someone could find this table by searching for its content

## Rules
- Be specific — mention actual values, not just "the table shows data"
- Keep it under 100 words
- Do NOT use bullet points — write flowing prose
- Output ONLY the summary text, no labels or formatting"""


def extract_tables_camelot(pdf_path: str, page_num: int) -> list[dict]:
    """
    Extracts tables from a specific page using Camelot.

    🧠 WHY PAGE-BY-PAGE:
    We already know which pages have tables from Phase 1's
    detect_table_regions() metadata. Processing only flagged pages
    is faster than scanning the whole PDF.

    Returns list of table dicts:
    [{
        "method":   "camelot_lattice" | "camelot_stream",
        "headers":  ["Term", "Meaning", "Use in Your Project"],
        "rows":     [["Bacteria", "Single-celled...", "Each genome..."]],
        "csv":      "Term,Meaning,...\nBacteria,...",
        "accuracy": 98.3,   ← Camelot's confidence score
    }]
    """
    results = []

    # Try lattice first (tables with visible borders)
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_num),
            flavor="lattice",
        )

        for table in tables:
            if table.accuracy < 50:
                # Low accuracy = likely a false positive detection
                print(f"    Camelot lattice: skipping low-accuracy table ({table.accuracy:.1f}%)")
                continue

            df      = table.df
            headers = df.iloc[0].tolist()   # first row = headers
            rows    = df.iloc[1:].values.tolist()
            csv_str = df.to_csv(index=False, header=False)

            results.append({
                "method":   "camelot_lattice",
                "headers":  headers,
                "rows":     rows,
                "csv":      csv_str,
                "accuracy": table.accuracy,
            })
            print(f"    Camelot lattice: found table ({table.accuracy:.1f}% accuracy, "
                  f"{len(rows)} rows × {len(headers)} cols)")

    except Exception as e:
        print(f"    Camelot lattice: failed — {e}")

    # If lattice found nothing, try stream (borderless tables)
    if not results:
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor="stream",
            )

            for table in tables:
                if table.accuracy < 40:
                    continue

                df      = table.df
                headers = df.iloc[0].tolist()
                rows    = df.iloc[1:].values.tolist()
                csv_str = df.to_csv(index=False, header=False)

                results.append({
                    "method":   "camelot_stream",
                    "headers":  headers,
                    "rows":     rows,
                    "csv":      csv_str,
                    "accuracy": table.accuracy,
                })
                print(f"    Camelot stream: found table ({table.accuracy:.1f}% accuracy, "
                      f"{len(rows)} rows × {len(headers)} cols)")

        except Exception as e:
            print(f"    Camelot stream: failed — {e}")

    return results


def extract_page_as_image(pdf_path: str, page_num: int, dpi: int = 200) -> bytes | None:
    """
    Renders a PDF page as a PNG image using PyMuPDF.

    🧠 CONCEPT: PyMuPDF page rendering
    fitz.open() loads the PDF. page.get_pixmap() renders the page
    to a raster image at the specified DPI.

    DPI 200 = good balance:
      - High enough for OCR/vision model to read text clearly
      - Low enough to keep image size reasonable (~300-500KB per page)
      DPI 300 = sharper but 2x larger files
      DPI 150 = faster but text may be blurry for vision model

    We return raw PNG bytes — not a file path — so we can encode
    directly to base64 without disk I/O.

    🧠 WHY PyMuPDF FOR RENDERING:
    pdfplumber can't render pages to images.
    PyMuPDF (fitz) is the best Python library for PDF rendering —
    fast, accurate, handles complex layouts.
    """
    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num - 1]  # fitz is 0-indexed, we use 1-indexed

        # Create transformation matrix for desired DPI
        # Default PDF resolution is 72 DPI
        # zoom = desired_dpi / 72
        zoom   = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(matrix=matrix)
        doc.close()

        return pixmap.tobytes("png")

    except Exception as e:
        print(f"    PyMuPDF render failed: {e}")
        return None


def extract_table_from_image_vision(
    image_bytes: bytes,
    page_num: int,
    source: str,
) -> dict | None:
    """
    Sends a page image to Groq vision model to extract table data.

    🧠 CONCEPT: Vision LLM for image tables
    When a table is embedded as a raster image, there's no text layer
    to extract. We must send the image to a model that can SEE it.

    The Groq vision API accepts images as base64-encoded strings
    embedded in the message content alongside text instructions.

    Message format for vision:
    {
        "role": "user",
        "content": [
            {"type": "text",       "text": "your prompt"},
            {"type": "image_url",  "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }

    The model reads both the text prompt AND the image simultaneously —
    this is "multimodal" inference. The vision model understands spatial
    relationships in the image: rows, columns, alignment, cell boundaries.

    Returns structured table dict or None if extraction fails.
    """
    try:
        # Encode image to base64
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: base64 encoding
        # base64.b64encode() converts binary bytes → ASCII string.
        # .decode("utf-8") converts the bytes object to a Python string.
        # The "data:image/png;base64," prefix is the standard data URI
        # format that tells the API what type of image follows.
        # ─────────────────────────────────────────────────────────────────
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        print(f"    Vision model: sending page {page_num} image to {VISION_MODEL}...")

        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TABLE_VISION_PROMPT,
                        },
                        {
                            "type":      "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        # ── Robust JSON extraction ────────────────────────────────────────────
        # Vision models often prepend natural language before JSON
        # e.g. "Here is the output:\n{...}"
        # We find the first { and last } to extract just the JSON object.
        start = raw.find("{")
        end   = raw.rfind("}") + 1

        if start == -1 or end == 0:
            print(f"  Vision model: no JSON found in response")
            print(f"  Raw response: {raw[:300]}")
            return None

        json_str = raw[start:end]
        data     = json.loads(json_str)

        return {
            "method":   "vision",
            "headers":  data.get("headers", []),
            "rows":     data.get("rows", []),
            "csv":      data.get("csv", ""),
            "summary":  data.get("summary", ""),
            "accuracy": 100.0,  # vision model returns full extraction
        }

    except json.JSONDecodeError as e:
        print(f"    Vision model: JSON parse error — {e}")
        print(f"    Raw response: {raw[:300]}")
        return None
    except Exception as e:
        print(f"    Vision model: failed — {e}")
        return None


def generate_table_summary(csv_data: str, page_num: int) -> str:
    """
    Generates a natural language summary of a Camelot-extracted table.

    🧠 WHY GENERATE SUMMARY FOR CAMELOT TABLES:
    Camelot gives us structured CSV data — perfect for BM25 exact matching.
    But vector search needs natural language to understand meaning.
    A query like "what does plasmid mean?" needs the summary to match —
    the CSV "Plasmid,Extra DNA in bacteria carrying genes,..." is less
    semantically rich for embedding models.

    We ask Groq to write a summary that:
    - Describes the table's topic
    - Mentions key values explicitly
    - Is searchable by both semantic and keyword queries

    Vision-extracted tables already include a summary from the vision model,
    so this function is only called for Camelot-extracted tables.
    """
    try:
        prompt = TABLE_SUMMARY_PROMPT.format(
            page=page_num,
            csv_data=csv_data[:2000],  # cap at 2000 chars to stay within token limit
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"    Summary generation failed: {e}")
        return f"Table extracted from page {page_num}."


def build_table_chunk(
    table:       dict,
    page:        dict,
    table_index: int,
) -> dict:
    """
    Converts an extracted table into a chunk dict for ChromaDB.

    🧠 CONCEPT: Hybrid content for table chunks
    The chunk "content" field combines:

    1. Natural language summary (top)
       → What gets embedded for semantic search
       → Vector search finds this when query asks about table topics

    2. Raw CSV data (bottom)
       → What BM25 indexes for keyword search
       → BM25 finds exact terms: "MIC", "K-mer", "FASTA", "76.1%"

    Both live in one chunk so a single retrieval hit gives the LLM
    both the semantic context AND the exact data to answer with.

    ID format: {doc_id}_table_p{page}_{index}
    Example:   970c8d73_table_p3_0
    """
    doc_id   = page["doc_id"]
    page_num = page["page"]
    source   = page["source"]

    # Generate summary if not already provided by vision model
    summary = table.get("summary") or generate_table_summary(
        table["csv"], page_num
    )

    # Build hybrid content: summary + raw CSV
    csv_preview = table["csv"][:1500]  # cap very large tables
    content = (
        f"{summary}\n\n"
        f"Raw table data:\n{csv_preview}"
    )

    return {
        "id":           f"{doc_id}_table_p{page_num}_{table_index}",
        "doc_id":       doc_id,
        "source":       source,
        "page":         page_num,
        "type":         "table",
        "method":       table["method"],
        "chunk_index":  table_index,
        "content":      content,
        "token_count":  len(content.split()),   # approximate
        "has_tables":   True,
        "headers":      str(table.get("headers", [])),
        "accuracy":     table.get("accuracy", 0),
    }


def extract_tables_from_pdf(pdf_path: str, pages: list[dict]) -> list[dict]:
    """
    Main table extraction pipeline for a full document.

    Strategy per page:
      1. If page has text layer → try Camelot (lattice → stream)
      2. If Camelot finds nothing OR page is image-only → try vision model
      3. Build chunk dicts from all extracted tables

    Args:
        pdf_path : path to the PDF file
        pages    : all page dicts from extract_text.py (including skipped)

    Returns: list of table chunk dicts ready for ChromaDB
    """
    all_table_chunks = []

    # Process pages flagged as having tables OR as image-only
    target_pages = [
        p for p in pages
        if p.get("has_tables") or p.get("skip_reason") == "no_text_layer"
    ]

    if not target_pages:
        print("  No pages flagged for table extraction.")
        return []

    print(f"  Processing {len(target_pages)} pages for tables...")

    for page in target_pages:
        page_num   = page["page"]
        is_skipped = page.get("skip_reason") == "no_text_layer"

        print(f"\n  Page {page_num} "
              f"({'image-only' if is_skipped else 'has table regions'}):")

        tables_found = []

        # ── Step 1: Try Camelot (only for pages with text layer) ──────────
        if not is_skipped:
            camelot_tables = extract_tables_camelot(pdf_path, page_num)
            tables_found.extend(camelot_tables)

        # ── Step 2: Vision model if Camelot found nothing or page is image ─
        if not tables_found:
            print(f"    Camelot found nothing — trying vision model")
            image_bytes = extract_page_as_image(pdf_path, page_num)

            if image_bytes:
                vision_table = extract_table_from_image_vision(
                    image_bytes, page_num, page["source"]
                )
                if vision_table:
                    tables_found.append(vision_table)

        # ── Step 3: Build chunk dicts ─────────────────────────────────────
        if tables_found:
            for i, table in enumerate(tables_found):
                chunk = build_table_chunk(table, page, i)
                all_table_chunks.append(chunk)
                print(f"    → Chunk created: {chunk['id']} "
                      f"({chunk['method']}, ~{chunk['token_count']} tokens)")
        else:
            print(f"    → No tables extracted from page {page_num}")

    print(f"\n  ✓ Total table chunks: {len(all_table_chunks)}")
    return all_table_chunks