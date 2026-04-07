# # src/ingest/extract_text.py

# import pdfplumber
# import json
# import hashlib
# from pathlib import Path


# def generate_doc_id(pdf_path: str) -> str:
#     """
#     Creates a unique ID for this document based on its filename.

#     WHY: When we have multiple PDFs in the vector DB, every chunk needs
#     to know which document it came from. We use an MD5 hash of the
#     filename so the ID is short, consistent, and reproducible.

#     ALTERNATIVE: You could use a UUID, but then the same file gets a
#     different ID every run — makes deduplication impossible.
#     """
#     return hashlib.md5(Path(pdf_path).name.encode()).hexdigest()[:8]


# def detect_table_regions(page) -> list[dict]:
#     """
#     Uses pdfplumber's built-in table detection to find table bounding
#     boxes on a page — WITHOUT extracting the table data yet.

#     WHY PRE-DETECT: In Phase 2 we'll extract tables properly with Camelot.
#     But Camelot needs to know WHICH pages have tables and WHERE they are.
#     Storing bounding boxes now means Phase 2 can skip full re-extraction.

#     HOW pdfplumber detects tables:
#     It looks for lines (horizontal + vertical rules) that form a grid
#     structure. Where lines intersect and form cells → likely a table.

#     LIMITATION: Tables without visible borders (styled with shading only)
#     are often missed. We'll handle that edge case in Phase 2.

#     Returns a list of bounding box dicts:
#     [{ "x0": 50, "y0": 200, "x1": 550, "y1": 400 }, ...]
#     """
#     tables = page.find_tables()  # pdfplumber's detection algorithm

#     regions = []
#     for table in tables:
#         bbox = table.bbox  # (x0, y0, x1, y1) in PDF coordinate units
#         regions.append({
#             "x0": round(bbox[0], 2),
#             "y0": round(bbox[1], 2),
#             "x1": round(bbox[2], 2),
#             "y1": round(bbox[3], 2),
#         })

#     return regions


# def extract_text_outside_tables(page, table_regions: list[dict]) -> str:
#     """
#     Extracts ONLY the text that sits outside detected table regions.

#     WHY: If we extract all text naively, table cells get mixed into
#     the paragraph text, creating garbled chunks like:
#     "The results showed Model A 92.3 Model B 94.1 significant improvement"

#     Instead we crop each table region OUT of the page before extracting.
#     Tables will be handled properly in Phase 2.

#     HOW pdfplumber cropping works:
#     page.crop(bbox) returns a new virtual page object containing only
#     the content within that bounding box. We do the inverse — extract
#     from everything EXCEPT those boxes.

#     ALTERNATIVE APPROACH: Extract everything, then post-process with
#     regex to remove table-like content. Messier and error-prone.
#     """
#     if not table_regions:
#         # No tables on this page — extract everything normally
#         return page.extract_text() or ""

#     # Build a list of "safe" vertical bands (regions without tables).
#     # pdfplumber coordinate system: y=0 is TOP of page (opposite of math).
#     page_height = page.height

#     # Collect all y-ranges occupied by tables
#     table_y_ranges = [(r["y0"], r["y1"]) for r in table_regions]

#     # We'll extract text from the full page then let pdfplumber's
#     # bounding box filtering handle exclusion
#     # Simpler and more reliable than manual band slicing
#     excluded_bboxes = [
#         (r["x0"], r["y0"], r["x1"], r["y1"]) for r in table_regions
#     ]

#     # Crop page to exclude each table region
#     # We collect words outside all table bounding boxes
#     words = page.extract_words()
#     clean_words = []

#     for word in words:
#         in_table = False
#         wx0, wy0, wx1, wy1 = word["x0"], word["top"], word["x1"], word["bottom"]

#         for (tx0, ty0, tx1, ty1) in excluded_bboxes:
#             # Check if word center falls inside a table region
#             word_cx = (wx0 + wx1) / 2
#             word_cy = (wy0 + wy1) / 2
#             if tx0 <= word_cx <= tx1 and ty0 <= word_cy <= ty1:
#                 in_table = True
#                 break

#         if not in_table:
#             clean_words.append(word["text"])

#     return " ".join(clean_words)


# def extract_pdf(pdf_path: str) -> list[dict]:
#     """
#     Main extraction function. Processes every page and returns:

#     [
#         {
#             "doc_id":        "a3f8c1d2",          # unique per document
#             "source":        "research_paper.pdf", # original filename
#             "page":          1,                    # 1-indexed
#             "text":          "Introduction ...",   # clean text, no table content
#             "char_count":    843,                  # useful for debugging thin pages
#             "has_tables":    True,                 # flag for Phase 2
#             "table_regions": [{ x0, y0, x1, y1 }],# bounding boxes for Phase 2
#             "width":         612.0,                # page dimensions (PDF points)
#             "height":        792.0,                # 72 points = 1 inch
#         },
#         ...
#     ]

#     WHY RETURN A LIST OF DICTS and not one big string?
#     Each page becomes its own unit. Page number travels with the text
#     all the way to the final answer → "Source: page 4 of research_paper.pdf"
#     """
#     doc_id = generate_doc_id(pdf_path)
#     source_name = Path(pdf_path).name
#     pages = []

#     print(f"\n📄 Extracting: {source_name}  (doc_id: {doc_id})")

#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
#         print(f"   Total pages: {total_pages}")

#         for page_num, page in enumerate(pdf.pages, start=1):

#             # ── Step 1: detect table regions ──────────────────────────────
#             table_regions = detect_table_regions(page)
#             has_tables = len(table_regions) > 0

#             if has_tables:
#                 print(f"   Page {page_num}: {len(table_regions)} table(s) detected")

#             # ── Step 2: extract text outside tables ────────────────────────
#             text = extract_text_outside_tables(page, table_regions)

#             # ── Step 3: skip pages with no useful content ──────────────────
#             # A page might be all images, a full-page table, or blank.
#             # char_count < 50 is a heuristic — tune this for your PDFs.
#             if not text or len(text.strip()) < 50:
#                 print(f"   Page {page_num}: skipped (image-only or blank)")
#                 continue

#             # ── Step 4: assemble the page record ───────────────────────────
#             pages.append({
#                 "doc_id":        doc_id,
#                 "source":        source_name,
#                 "page":          page_num,
#                 "text":          text.strip(),
#                 "char_count":    len(text.strip()),
#                 "has_tables":    has_tables,
#                 "table_regions": table_regions,
#                 "width":         round(page.width, 2),
#                 "height":        round(page.height, 2),
#             })

#     print(f"\n✓ Done — {len(pages)} pages extracted")
#     print(f"  Pages with tables: {sum(1 for p in pages if p['has_tables'])}")
#     print(f"  Total characters: {sum(p['char_count'] for p in pages):,}")

#     return pages


# def save_extracted(pages: list[dict], output_dir: str = "data/extracted/text"):
#     """
#     Saves the extracted pages to JSON.

#     WHY JSON and not a database yet?
#     At this stage we want to inspect the output easily — open the file,
#     read it, spot problems. Once the pipeline is proven we can swap this
#     for a proper store. JSON is the right tool for prototyping.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # One JSON file per document, named by doc_id
#     doc_id = pages[0]["doc_id"]
#     source = pages[0]["source"].replace(".pdf", "")
#     output_path = output_dir / f"{source}_{doc_id}.json"

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(pages, f, indent=2, ensure_ascii=False)

#     print(f"✓ Saved → {output_path}")
#     return str(output_path)


# # ── Quick test runner ────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python -m src.ingest.extract_text <path_to_pdf>")
#         sys.exit(1)

#     pdf_path = sys.argv[1]

#     # Extract
#     pages = extract_pdf(pdf_path)

#     # Preview first page
#     if pages:
#         print("\n── First page preview ──────────────────────────────────")
#         print(f"doc_id      : {pages[0]['doc_id']}")
#         print(f"source      : {pages[0]['source']}")
#         print(f"page        : {pages[0]['page']}")
#         print(f"has_tables  : {pages[0]['has_tables']}")
#         print(f"char_count  : {pages[0]['char_count']}")
#         print(f"text preview: {pages[0]['text'][:300]}")

#     # Save
#     save_extracted(pages)

# src/ingest/extract_text.py

import pdfplumber
import json
import hashlib
from pathlib import Path


def generate_doc_id(pdf_path: str) -> str:
    """
    Stable unique ID for a document based on its filename.
    MD5 of filename → first 8 hex chars.
    Same file always gets same ID → safe to re-run without duplicating.
    """
    return hashlib.md5(Path(pdf_path).name.encode()).hexdigest()[:8]


def detect_table_regions(page) -> list[dict]:
    """
    Detects table bounding boxes on a page using pdfplumber's line-grid detection.

    IMPORTANT — what we changed and why:
    Previously we used these regions to FILTER text out of extraction.
    That was wrong — pdfplumber over-detects tables on bullet lists and
    structured text, causing real content to be silently removed.

    Now we ONLY store the regions as metadata.
    Phase 2 (Camelot) will use them to locate and extract actual tables.
    Phase 1 text extraction ignores them completely.

    Returns [] if no tables found, or list of bounding box dicts.
    """
    try:
        tables = page.find_tables()
        regions = []
        for table in tables:
            bbox = table.bbox  # (x0, y0, x1, y1)
            regions.append({
                "x0": round(bbox[0], 2),
                "y0": round(bbox[1], 2),
                "x1": round(bbox[2], 2),
                "y1": round(bbox[3], 2),
            })
        return regions

    except Exception:
        # pdfplumber occasionally throws on malformed PDFs
        # Safe to skip — we just won't have table metadata for this page
        return []


def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts ALL text from every page plus table location metadata.

    Key design decision:
    We extract text unconditionally with page.extract_text().
    Table regions are stored as metadata ONLY — they do not affect
    what text gets extracted in Phase 1.

    Returns a list of page dicts:
    [
        {
            "doc_id":        "970c8d73",
            "source":        "Antibiotic_resistance.pdf",
            "page":          1,
            "text":          "Antibiotic Resistance Prediction...",
            "char_count":    1842,
            "has_tables":    False,
            "table_regions": [],
            "width":         612.0,
            "height":        792.0,
        },
        ...
    ]
    """
    doc_id = generate_doc_id(pdf_path)
    source_name = Path(pdf_path).name
    pages = []

    print(f"\n📄 Extracting: {source_name}  (doc_id: {doc_id})")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Total pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, start=1):

            # ── Step 1: extract all text unconditionally ───────────────────
            # extract_text() returns a single string for the page, or None
            # if the page contains no extractable text (scanned image page).
            text = page.extract_text()

            # ── Step 2: detect table regions (metadata only) ───────────────
            # Stored so Phase 2 knows which pages/regions to send to Camelot.
            # Has zero effect on the text extracted above.
            table_regions = detect_table_regions(page)
            has_tables = len(table_regions) > 0

            if has_tables:
                print(f"   Page {page_num}: {len(table_regions)} table region(s) flagged for Phase 2")

            # ── Step 3: skip genuinely empty pages ─────────────────────────
            # A page with no extractable text is either:
            #   a) a scanned image (no text layer) → handled in Phase 3
            #   b) a purely decorative page (divider, blank)
            # Threshold of 30 chars catches near-empty pages safely.
            # Lower than before (was 50) because short pages with just a
            # title or section heading are still worth keeping.
            if not text or len(text.strip()) < 30:
                print(f"   Page {page_num}: skipped — no extractable text (image-only or blank)")
                # Store a record anyway so Phase 3 knows to look for images here
                pages.append({
                    "doc_id":        doc_id,
                    "source":        source_name,
                    "page":          page_num,
                    "text":          "",
                    "char_count":    0,
                    "has_tables":    has_tables,
                    "table_regions": table_regions,
                    "width":         round(page.width, 2),
                    "height":        round(page.height, 2),
                    "skip_reason":   "no_text_layer",   # Phase 3 uses this flag
                })
                continue

            # ── Step 4: store the page ─────────────────────────────────────
            pages.append({
                "doc_id":        doc_id,
                "source":        source_name,
                "page":          page_num,
                "text":          text.strip(),
                "char_count":    len(text.strip()),
                "has_tables":    has_tables,
                "table_regions": table_regions,
                "width":         round(page.width, 2),
                "height":        round(page.height, 2),
                "skip_reason":   None,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    extractable = [p for p in pages if p["char_count"] > 0]
    skipped     = [p for p in pages if p["char_count"] == 0]
    table_pages = [p for p in pages if p["has_tables"]]

    print(f"\n✓ Done")
    print(f"  Extractable pages : {len(extractable)}")
    print(f"  Skipped pages     : {len(skipped)} (image-only — queued for Phase 3)")
    print(f"  Pages with tables : {len(table_pages)} (queued for Phase 2)")
    print(f"  Total characters  : {sum(p['char_count'] for p in pages):,}")

    return pages


def save_extracted(pages: list[dict], output_dir: str = "data/extracted/text"):
    """
    Saves ALL pages (including skipped ones) to JSON.

    Why save skipped pages too?
    Phase 3 needs to know which pages had no text layer so it can
    run image extraction on exactly those pages — no re-scan needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_id  = pages[0]["doc_id"]
    source  = pages[0]["source"].replace(".pdf", "")
    out_path = output_dir / f"{source}_{doc_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved → {out_path}")
    return str(out_path)


# ── Quick test runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.extract_text <path_to_pdf>")
        sys.exit(1)

    pages = extract_pdf(sys.argv[1])

    # Preview each extractable page
    print("\n── Page previews ────────────────────────────────────────────")
    for p in pages:
        if p["char_count"] > 0:
            print(f"\n  Page {p['page']}  |  {p['char_count']} chars  |  tables: {p['has_tables']}")
            print(f"  {p['text'][:200]}...")

    save_extracted(pages)