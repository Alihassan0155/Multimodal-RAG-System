# main.py

"""
Multimodal RAG — single entry point.

Usage:
  python main.py ingest  <pdf_path>       Ingest a PDF into ChromaDB
  python main.py stats                    Show what's currently in ChromaDB
  python main.py delete  <doc_id>         Remove a document from ChromaDB
  python main.py query   "<question>"     Query the RAG system (Phase 2)
"""

import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: why lazy imports inside each command function?
#
# If we import everything at the top of main.py, the embedding model
# (SentenceTransformer) loads into RAM every single time you run ANY command —
# even `python main.py stats` which doesn't need the model at all.
#
# Lazy imports (inside each function) mean:
#   - `stats` and `delete` are near-instant — no model loaded
#   - `ingest` and `query` load the model only when actually needed
#
# This is a common pattern in CLI tools. Same reason git doesn't load
# your entire repo into RAM just to run `git status`.
# ─────────────────────────────────────────────────────────────────────────────


def cmd_ingest(pdf_path: str):
    """
    Runs the full ingest pipeline on a PDF:
    extract → chunk → embed → store in ChromaDB.
    """
    from src.pipeline.ingest_pipeline import ingest

    path = Path(pdf_path)

    if not path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    if path.suffix.lower() != ".pdf":
        print(f"❌ Expected a .pdf file, got: {path.suffix}")
        sys.exit(1)

    ingest(str(path))


def cmd_stats():
    """
    Shows what's currently stored in ChromaDB.
    Does NOT load the embedding model — fast to run.
    """
    import chromadb

    print("\n📊 ChromaDB stats")
    print("─" * 40)

    try:
        client     = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("documents")
        total      = collection.count()

        if total == 0:
            print("  Collection is empty — run `ingest` first.")
            return

        results = collection.get(include=["metadatas"])
        metas   = results["metadatas"]

        # Aggregate per document
        docs: dict[str, dict] = {}
        for m in metas:
            did = m["doc_id"]
            if did not in docs:
                docs[did] = {
                    "source":      m["source"],
                    "doc_id":      did,
                    "chunks":      0,
                    "pages":       set(),
                    "has_tables":  False,
                }
            docs[did]["chunks"] += 1
            docs[did]["pages"].add(m["page"])
            if m.get("has_tables"):
                docs[did]["has_tables"] = True

        print(f"  Total chunks    : {total}")
        print(f"  Total documents : {len(docs)}\n")

        for doc in docs.values():
            print(f"  📄 {doc['source']}")
            print(f"     doc_id  : {doc['doc_id']}")
            print(f"     chunks  : {doc['chunks']}")
            print(f"     pages   : {sorted(doc['pages'])}")
            print(f"     tables  : {'yes — queued for Phase 2' if doc['has_tables'] else 'none detected'}")
            print()

    except Exception as e:
        print(f"  ❌ Could not read ChromaDB: {e}")
        print("  Have you ingested any documents yet?")


def cmd_delete(doc_id: str):
    """
    Deletes all chunks for a given doc_id from ChromaDB.
    Use this before re-ingesting an updated PDF.
    Does NOT load the embedding model — fast to run.
    """
    import chromadb

    print(f"\n🗑  Deleting doc_id: {doc_id}")
    print("─" * 40)

    try:
        client     = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("documents")

        # Check it exists first
        results = collection.get(
            where={"doc_id": {"$eq": doc_id}},
            limit=1,
        )

        if not results["ids"]:
            print(f"  ⚠ No document found with doc_id '{doc_id}'")
            print("  Run `python main.py stats` to see available doc_ids.")
            return

        collection.delete(where={"doc_id": {"$eq": doc_id}})
        print(f"  ✓ Deleted all chunks for '{doc_id}'")
        print(f"  Remaining chunks: {collection.count()}")

    except Exception as e:
        print(f"  ❌ Error: {e}")



    """
    Query the RAG system with a natural language question.

    ── Phase 1 (now): vector-only search, returns raw chunks ──────────────
    ── Phase 2 (next): hybrid BM25 + vector + RRF + reranker + LLM answer ─

    Right now this does a basic vector search so you can verify the
    pipeline works end to end before we build the full retriever.
    """
    import chromadb
    from src.pipeline.vectorstore import model, EMBEDDING_MODEL

    print(f"\n🔍 Query: {question}")
    print("─" * 40)

    try:
        client     = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("documents")

        if collection.count() == 0:
            print("  ⚠ ChromaDB is empty — run `ingest` first.")
            return

        # Embed the question using the same model used at ingest time
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: query-time embedding
        # The query MUST be embedded with the exact same model as the chunks.
        # Different models produce different vector spaces — mixing them
        # means your query vector and chunk vectors aren't comparable.
        # This is why the embedding model is a pipeline-level constant,
        # not something you change per query.
        # ─────────────────────────────────────────────────────────────────
        print(f"  Embedding question with {EMBEDDING_MODEL}...")
        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=False,
        )[0]

        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()

        # Vector search — top 3 most similar chunks
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: distance vs similarity
        # ChromaDB returns "distances" not "similarities".
        # With cosine metric: distance = 1 - cosine_similarity
        # So distance 0.0 = perfect match, distance 1.0 = no relation.
        # We convert to similarity = 1 - distance for readability.
        # ─────────────────────────────────────────────────────────────────
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\n  Top {len(docs)} results:\n")

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = round(1 - dist, 3)
            print(f"  [{i}] similarity: {similarity}  |  "
                  f"source: {meta['source']}  |  "
                  f"page: {meta['page']}")
            print(f"      {doc[:200]}...")
            print()

        print("  ⚠ Full LLM answering coming in Phase 2 (retriever + reranker + Claude)")

    except Exception as e:
        print(f"  ❌ Error: {e}")
# def cmd_query(question: str):
    """
    Full RAG pipeline:
    Query expansion → Multi-query hybrid retrieval → Rerank → Gemini answer
    """
    from src.retrieval.retriever import HybridRetriever
    from src.pipeline.answerer import expand_query, answer

    print(f"\n🔍 Query: {question}")
    print("─" * 55)

    try:
        # ── Step 1: Query expansion ───────────────────────────────────────
        print("\n[1/3] Expanding query...")
        expanded_queries = expand_query(question)

        # ── Step 2: Multi-query hybrid retrieval ──────────────────────────
        print("\n[2/3] Retrieving...")
        retriever = HybridRetriever(top_candidates=20)
        chunks    = retriever.retrieve_multi(
            queries        = expanded_queries,
            original_query = question,
            top_k          = 5,
        )

        if not chunks:
            print("\n  ℹ No relevant chunks found.")
            print("  This question appears to be outside the document scope.")
            return

        # ── Step 3: Generate answer ───────────────────────────────────────
        print("\n[3/3] Generating answer...")
        response = answer(question, chunks)

        # ── Display ───────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  ANSWER")
        print("=" * 55)
        print(response)
        print("=" * 55)
        print(f"\n  Sources used:")
        for i, chunk in enumerate(chunks, 1):
            print(f"    [{i}] {chunk['source']} — page {chunk['page']} "
                  f"(rerank: {chunk['rerank_score']:.2f})")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise
def cmd_query(question: str):
    """
    Full RAG pipeline:
    Intent classify → Query expand → Multi-query retrieve
    → Graph expand → Rerank → Groq answer
    """
    from src.pipeline.query_classifier import classify_query
    from src.retrieval.retriever import HybridRetriever
    from src.pipeline.answerer import expand_query, answer, ANSWERING_PROMPT, build_context_block

    print(f"\n🔍 Query: {question}")
    print("─" * 55)

    try:
        # ── Step 0: Classify intent ───────────────────────────────────────
        print("\n[0/3] Classifying query intent...")
        intent_config = classify_query(question)

        top_k          = intent_config["top_k"]
        expand_enabled = intent_config["expand_queries"]
        llm_instruction = intent_config["llm_instruction"]

        print(f"  Intent   : {intent_config['intent']}")
        print(f"  top_k    : {top_k}")
        print(f"  Threshold: {intent_config['threshold']}")

        # ── Step 1: Query expansion ───────────────────────────────────────
        if expand_enabled:
            print("\n[1/3] Expanding query...")
            expanded_queries = expand_query(question)
        else:
            print("\n[1/3] Skipping expansion (boolean query)")
            expanded_queries = [question]

        # ── Step 2: Retrieve + graph expand ──────────────────────────────
        print("\n[2/3] Retrieving...")
        retriever = HybridRetriever(top_candidates=20)
        chunks    = retriever.retrieve_multi(
            queries         = expanded_queries,
            original_query  = question,
            top_k           = top_k,
            intent_threshold = intent_config["threshold"],
        )

        if not chunks:
            print("\n  ℹ No relevant chunks found.")
            print("  This question appears to be outside the document scope.")
            return

        # ── Step 3: Generate answer ───────────────────────────────────────
        print("\n[3/3] Generating answer...")
        response = answer(question, chunks, llm_instruction)

        # ── Display ───────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  ANSWER  [{intent_config['intent'].upper()}]")
        print("=" * 55)
        print(response)
        print("=" * 55)
        print(f"\n  Sources used ({len(chunks)} chunks):")
        for i, chunk in enumerate(chunks, 1):
            graph_tag = f" [graph:{chunk.get('graph_edge','─')}]" \
                        if chunk.get("graph_edge") else ""
            print(f"    [{i}] {chunk['source']} — page {chunk['page']} "
                  f"(rerank: {chunk.get('rerank_score', 0):.2f})"
                  f"{graph_tag}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise
# ─────────────────────────────────────────────────────────────────────────────
# Command dispatcher
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    "ingest": {
        "fn":      cmd_ingest,
        "args":    "<pdf_path>",
        "desc":    "Ingest a PDF into ChromaDB",
        "example": "python main.py ingest data/pdfs/paper.pdf",
    },
    "stats": {
        "fn":      cmd_stats,
        "args":    "",
        "desc":    "Show what's currently in ChromaDB",
        "example": "python main.py stats",
    },
    "delete": {
        "fn":      cmd_delete,
        "args":    "<doc_id>",
        "desc":    "Remove a document from ChromaDB",
        "example": "python main.py delete 970c8d73",
    },
    "query": {
        "fn":      cmd_query,
        "args":    '"<question>"',
        "desc":    "Query the RAG system",
        "example": 'python main.py query "what is antibiotic resistance?"',
    },
}


def print_help():
    print("\n🧠 Multimodal RAG — command reference")
    print("─" * 45)
    for name, info in COMMANDS.items():
        args = f" {info['args']}" if info["args"] else ""
        print(f"  python main.py {name}{args}")
        print(f"    {info['desc']}")
        print(f"    e.g. {info['example']}\n")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command not in COMMANDS:
        print(f"\n❌ Unknown command: '{command}'")
        print_help()
        sys.exit(1)

    # Route to the right function with remaining args
    if command == "ingest":
        if len(sys.argv) < 3:
            print("❌ Missing pdf_path. Example:")
            print("   python main.py ingest data/pdfs/paper.pdf")
            sys.exit(1)
        cmd_ingest(sys.argv[2])

    elif command == "stats":
        cmd_stats()

    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Missing doc_id. Example:")
            print("   python main.py delete 970c8d73")
            sys.exit(1)
        cmd_delete(sys.argv[2])

    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Missing question. Example:")
            print('   python main.py query "what is antibiotic resistance?"')
            sys.exit(1)
        # Join remaining args in case question has spaces without quotes
        question = " ".join(sys.argv[2:])
        cmd_query(question)


if __name__ == "__main__":
    main()