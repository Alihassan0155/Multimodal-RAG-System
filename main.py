
# """
# Multimodal RAG — single entry point.

# Usage:
#   python main.py ingest  <pdf_path>
#   python main.py stats
#   python main.py delete  <doc_id>
#   python main.py query   "<question>"
# """

# import sys
# from pathlib import Path


# # ─────────────────────────────────────────────────────────────
# # INGEST
# # ─────────────────────────────────────────────────────────────
# def cmd_ingest(pdf_path: str):
#     from src.pipeline.ingest_pipeline import ingest

#     path = Path(pdf_path)

#     if not path.exists():
#         print(f"❌ File not found: {pdf_path}")
#         sys.exit(1)

#     if path.suffix.lower() != ".pdf":
#         print(f"❌ Expected a .pdf file, got: {path.suffix}")
#         sys.exit(1)

#     ingest(str(path))


# # ─────────────────────────────────────────────────────────────
# # STATS
# # ─────────────────────────────────────────────────────────────
# def cmd_stats():
#     import chromadb

#     print("\n📊 ChromaDB stats")
#     print("─" * 40)

#     try:
#         client     = chromadb.PersistentClient(path="chroma_db")
#         collection = client.get_or_create_collection("documents")
#         total      = collection.count()

#         if total == 0:
#             print("  Collection is empty — run `ingest` first.")
#             return

#         results = collection.get(include=["metadatas"])
#         metas   = results["metadatas"]

#         docs = {}
#         for m in metas:
#             did = m["doc_id"]
#             if did not in docs:
#                 docs[did] = {
#                     "source": m["source"],
#                     "chunks": 0,
#                     "pages": set(),
#                 }
#             docs[did]["chunks"] += 1
#             docs[did]["pages"].add(m["page"])

#         print(f"  Total chunks    : {total}")
#         print(f"  Total documents : {len(docs)}\n")

#         for doc in docs.values():
#             print(f"  📄 {doc['source']}")
#             print(f"     chunks  : {doc['chunks']}")
#             print(f"     pages   : {sorted(doc['pages'])}")
#             print()

#     except Exception as e:
#         print(f"  ❌ Could not read ChromaDB: {e}")


# # ─────────────────────────────────────────────────────────────
# # DELETE
# # ─────────────────────────────────────────────────────────────
# def cmd_delete(doc_id: str):
#     import chromadb

#     print(f"\n🗑  Deleting doc_id: {doc_id}")
#     print("─" * 40)

#     try:
#         client     = chromadb.PersistentClient(path="chroma_db")
#         collection = client.get_or_create_collection("documents")

#         results = collection.get(where={"doc_id": {"$eq": doc_id}}, limit=1)

#         if not results["ids"]:
#             print(f"  ⚠ No document found with doc_id '{doc_id}'")
#             return

#         collection.delete(where={"doc_id": {"$eq": doc_id}})
#         print(f"  ✓ Deleted")
#         print(f"  Remaining chunks: {collection.count()}")

#     except Exception as e:
#         print(f"  ❌ Error: {e}")


# # ─────────────────────────────────────────────────────────────
# # QUERY (🔥 FULL PIPELINE)
# # ─────────────────────────────────────────────────────────────
# def cmd_query(question: str):
#     """
#     Full RAG pipeline:
#     Intent → Expand → Retrieve → Rerank → Answer
#     """

#     from src.pipeline.query_classifier import classify_query
#     from src.retrieval.retriever import HybridRetriever
#     from src.pipeline.answerer import expand_query, answer

#     print(f"\n🔍 Query: {question}")
#     print("─" * 55)

#     try:
#         # ── Step 0: Intent ─────────────────────────────────────
#         print("\n[0/3] Classifying query intent...")
#         intent_config = classify_query(question)

#         intent         = intent_config["intent"]
#         top_k          = intent_config["top_k"]
#         expand_enabled = intent_config["expand_queries"]
#         threshold      = intent_config["threshold"]

#         print(f"  Intent   : {intent}")
#         print(f"  top_k    : {top_k}")
#         print(f"  Threshold: {threshold}")

#         # ── Step 1: Expansion ──────────────────────────────────
#         if expand_enabled:
#             print("\n[1/3] Expanding query...")
#             expanded_queries = expand_query(question)
#         else:
#             print("\n[1/3] Skipping expansion (boolean query)")
#             expanded_queries = [question]

#         # ── Step 2: Retrieval ─────────────────────────────────
#         print("\n[2/3] Retrieving...")

#         retriever = HybridRetriever(top_candidates=20)

#         chunks = retriever.retrieve_multi(
#             queries          = expanded_queries,
#             original_query   = question,
#             top_k            = top_k,
#             intent_threshold = threshold,
#         )

#         if not chunks:
#             print("\n  ℹ No relevant chunks found.")
#             print("  This question appears to be outside the document scope.")
#             return

#         # ── Step 3: Answer ────────────────────────────────────
#         print("\n[3/3] Generating answer...")

#         response = answer(
#             query=question,
#             chunks=chunks,
#             intent=intent   # 🔥 FIX (CRITICAL)
#         )

#         # ── Display ───────────────────────────────────────────
#         print("\n" + "=" * 55)
#         print(f"  ANSWER  [{intent.upper()}]")
#         print("=" * 55)
#         print(response)
#         print("=" * 55)

#         print(f"\n  Sources used ({len(chunks)} chunks):")
#         for i, chunk in enumerate(chunks, 1):
#             print(
#                 f"    [{i}] {chunk['source']} — page {chunk['page']} "
#                 f"(rerank: {chunk.get('rerank_score', 0):.2f})"
#             )

#     except Exception as e:
#         print(f"  ❌ Error: {e}")
#         raise


# # ─────────────────────────────────────────────────────────────
# # COMMAND DISPATCH
# # ─────────────────────────────────────────────────────────────
# COMMANDS = {
#     "ingest": cmd_ingest,
#     "stats": cmd_stats,
#     "delete": cmd_delete,
#     "query": cmd_query,
# }


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python main.py [ingest|stats|delete|query]")
#         sys.exit(0)

#     command = sys.argv[1].lower()

#     if command not in COMMANDS:
#         print(f"❌ Unknown command: {command}")
#         sys.exit(1)

#     if command == "ingest":
#         cmd_ingest(sys.argv[2])

#     elif command == "stats":
#         cmd_stats()

#     elif command == "delete":
#         cmd_delete(sys.argv[2])

#     elif command == "query":
#         question = " ".join(sys.argv[2:])
#         cmd_query(question)


# if __name__ == "__main__":
#     main()

# main.py

"""
Multimodal RAG — single entry point.

Usage:
  python main.py ingest  <pdf_path>
  python main.py stats
  python main.py delete  <doc_id>
  python main.py query   "<question>"
"""

import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────
def cmd_ingest(pdf_path: str):
    from src.pipeline.ingest_pipeline import ingest

    path = Path(pdf_path)

    if not path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    if path.suffix.lower() != ".pdf":
        print(f"❌ Expected a .pdf file, got: {path.suffix}")
        sys.exit(1)

    ingest(str(path))


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────
def cmd_stats():
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

        docs = {}
        for m in metas:
            did = m["doc_id"]
            if did not in docs:
                docs[did] = {
                    "source": m["source"],
                    "chunks": 0,
                    "pages": set(),
                }
            docs[did]["chunks"] += 1
            docs[did]["pages"].add(m["page"])

        print(f"  Total chunks    : {total}")
        print(f"  Total documents : {len(docs)}\n")

        for doc in docs.values():
            print(f"  📄 {doc['source']}")
            print(f"     chunks  : {doc['chunks']}")
            print(f"     pages   : {sorted(doc['pages'])}")
            print()

    except Exception as e:
        print(f"  ❌ Could not read ChromaDB: {e}")


# ─────────────────────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────────────────────
def cmd_delete(doc_id: str):
    import chromadb

    print(f"\n🗑  Deleting doc_id: {doc_id}")
    print("─" * 40)

    try:
        client     = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("documents")

        results = collection.get(where={"doc_id": {"$eq": doc_id}}, limit=1)

        if not results["ids"]:
            print(f"  ⚠ No document found with doc_id '{doc_id}'")
            return

        collection.delete(where={"doc_id": {"$eq": doc_id}})
        print(f"  ✓ Deleted")
        print(f"  Remaining chunks: {collection.count()}")

    except Exception as e:
        print(f"  ❌ Error: {e}")


# ─────────────────────────────────────────────────────────────
# QUERY (🔥 FULL PIPELINE)
# ─────────────────────────────────────────────────────────────
def cmd_query(question: str):
    """
    Full RAG pipeline:
    Intent → Expand → Retrieve → Rerank → Answer
    """

    from src.pipeline.query_classifier import classify_query
    from src.retrieval.retriever import HybridRetriever
    from src.pipeline.answerer import expand_query, answer

    print(f"\n🔍 Query: {question}")
    print("─" * 55)

    try:
        # ── Step 0: Intent ─────────────────────────────────────
        print("\n[0/3] Classifying query intent...")
        intent_config = classify_query(question)

        intent         = intent_config["intent"]
        top_k          = intent_config["top_k"]
        expand_enabled = intent_config["expand_queries"]
        threshold      = intent_config["threshold"]

        print(f"  Intent   : {intent}")
        print(f"  top_k    : {top_k}")
        print(f"  Threshold: {threshold}")

        # ── Step 1: Expansion ──────────────────────────────────
        if expand_enabled:
            print("\n[1/3] Expanding query...")
            expanded_queries = expand_query(question)
        else:
            print("\n[1/3] Skipping expansion (boolean query)")
            expanded_queries = [question]

        # ── Step 2: Retrieval ─────────────────────────────────
        print("\n[2/3] Retrieving...")

        retriever = HybridRetriever(top_candidates=20)

        chunks = retriever.retrieve_multi(
            queries          = expanded_queries,
            original_query   = question,
            top_k            = top_k,
            intent_threshold = threshold,
        )

        if not chunks:
            print("\n  ℹ No relevant chunks found.")
            print("  This question appears to be outside the document scope.")
            return

        # ── Step 3: Answer ────────────────────────────────────
        print("\n[3/3] Generating answer...")

        response = answer(
            query=question,
            chunks=chunks,
            intent=intent   # 🔥 FIX (CRITICAL)
        )

        # ── Display ───────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  ANSWER  [{intent.upper()}]")
        print("=" * 55)
        print(response)
        print("=" * 55)

        print(f"\n  Sources used ({len(chunks)} chunks):")
        for i, chunk in enumerate(chunks, 1):
            print(
                f"    [{i}] {chunk['source']} — page {chunk['page']} "
                f"(rerank: {chunk.get('rerank_score', 0):.2f})"
            )

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# COMMAND DISPATCH
# ─────────────────────────────────────────────────────────────
COMMANDS = {
    "ingest": cmd_ingest,
    "stats": cmd_stats,
    "delete": cmd_delete,
    "query": cmd_query,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest|stats|delete|query]")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command not in COMMANDS:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

    if command == "ingest":
        cmd_ingest(sys.argv[2])

    elif command == "stats":
        cmd_stats()

    elif command == "delete":
        cmd_delete(sys.argv[2])

    elif command == "query":
        question = " ".join(sys.argv[2:])
        cmd_query(question)


if __name__ == "__main__":
    main()