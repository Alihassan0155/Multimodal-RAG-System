# # src/pipeline/ingest_pipeline.py

# from pathlib import Path
# from src.ingest.extract_text import extract_pdf
# from src.pipeline.chunker import chunk_document
# from src.pipeline.vectorstore import (
#     get_or_create_collection,
#     add_chunks_to_collection,
#     get_collection_stats,
# )


# def ingest(pdf_path: str):
#     print("=" * 55)
#     print(f"  INGESTING: {Path(pdf_path).name}")
#     print("=" * 55)

#     # Step 1: Extract
#     print("\n[1/3] Extracting text...")
#     pages      = extract_pdf(pdf_path)
#     text_pages = [p for p in pages if p["char_count"] > 0]

#     print(f"  {len(text_pages)} pages with text | "
#           f"{len(pages) - len(text_pages)} image-only (Phase 3)")

#     if not text_pages:
#         print("⚠ No extractable text found. Is this a scanned PDF?")
#         return

#     # Step 2: Chunk
#     print("\n[2/3] Chunking...")
#     chunks       = chunk_document(text_pages)
#     token_counts = [c["token_count"] for c in chunks]

#     print(f"  {len(chunks)} chunks created")
#     print(f"  Tokens — min: {min(token_counts)} | "
#           f"max: {max(token_counts)} | "
#           f"avg: {sum(token_counts) // len(token_counts)}")

#     # Step 3: Embed + Store
#     print("\n[3/3] Embedding and storing in ChromaDB...")
#     collection = get_or_create_collection("documents")
#     added      = add_chunks_to_collection(collection, chunks)

#     # Summary
#     print("\n" + "=" * 55)
#     stats = get_collection_stats(collection)
#     print(f"  DONE — {added} chunks added")
#     print(f"  ChromaDB total : {stats['total_chunks']} chunks "
#           f"across {stats['total_documents']} document(s)")
#     print(f"  Sources        : {', '.join(stats['sources'])}")
#     print("=" * 55)

# src/pipeline/ingest_pipeline.py

from pathlib import Path
from src.ingest.extract_text import extract_pdf
from src.ingest.extract_tables import extract_tables_from_pdf
from src.pipeline.chunker import chunk_document
from src.pipeline.vectorstore import (
    get_or_create_collection,
    add_chunks_to_collection,
    get_collection_stats,
)


def ingest(pdf_path: str):
    """
    Full pipeline: PDF → extract text → extract tables → chunk → embed → ChromaDB.
    """
    print("=" * 55)
    print(f"  INGESTING: {Path(pdf_path).name}")
    print("=" * 55)

    # ── Step 1: Extract text ──────────────────────────────────────────────
    print("\n[1/4] Extracting text...")
    pages      = extract_pdf(pdf_path)
    text_pages = [p for p in pages if p["char_count"] > 0]

    print(f"  {len(text_pages)} pages with text | "
          f"{len(pages) - len(text_pages)} image-only (Phase 3)")

    # ── Step 2: Extract tables ────────────────────────────────────────────
    print("\n[2/4] Extracting tables...")
    table_chunks = extract_tables_from_pdf(pdf_path, pages)
    print(f"  {len(table_chunks)} table chunks extracted")

    # ── Step 3: Chunk text pages ──────────────────────────────────────────
    print("\n[3/4] Chunking text...")
    text_chunks = chunk_document(text_pages) if text_pages else []

    # Combine text + table chunks
    all_chunks   = text_chunks + table_chunks
    token_counts = [c.get("token_count", 0) for c in text_chunks]

    print(f"  {len(text_chunks)} text chunks")
    print(f"  {len(table_chunks)} table chunks")
    print(f"  {len(all_chunks)} total chunks")

    if not all_chunks:
        print("⚠ No chunks created.")
        return

    # ── Step 4: Embed + Store ─────────────────────────────────────────────
    print("\n[4/4] Embedding and storing in ChromaDB...")
    collection = get_or_create_collection("documents")
    added      = add_chunks_to_collection(collection, all_chunks)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    stats = get_collection_stats(collection)
    print(f"  DONE — {added} chunks added")
    print(f"  ChromaDB total : {stats['total_chunks']} chunks "
          f"across {stats['total_documents']} document(s)")
    print(f"  Sources        : {', '.join(stats['sources'])}")
    print("=" * 55)