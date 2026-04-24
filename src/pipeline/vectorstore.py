# src/pipeline/vectorstore.py

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: SentenceTransformer
#
# sentence-transformers is a library that wraps HuggingFace models and
# makes them easy to use for embedding. One line to load, one line to embed.
#
# How it works internally:
#   text → tokenizer → transformer model → mean pooling → vector
#
# "Mean pooling" means: run the text through the model, get a vector for
# every token, then average them all into one single vector.
# That averaged vector represents the whole sentence/chunk.
#
# all-mpnet-base-v2 facts:
#   - Output: 768-dimensional vector
#   - Max input: 384 tokens (longer text gets truncated — important!)
#   - Size: ~420MB, downloaded once and cached in ~/.cache/huggingface/
#   - Speed: ~500 chunks/min on CPU — fast enough for our use case
#
# The 384-token limit is why our 512-token chunks are slightly oversized
# for this model. We have two options:
#   a) Accept truncation (last ~128 tokens of large chunks are ignored)
#   b) Reduce chunk size to 384 to match the model's window
#
# We'll go with 384 tokens per chunk when we re-run the chunker.
# I'll flag this after the vectorstore is confirmed working.
#
# Alternatives:
#   - BAAI/bge-large-en-v1.5  : 1024 dims, 512 token limit, ~1.3GB
#   - all-MiniLM-L6-v2        : 384 dims, 256 token limit, ~80MB
#   - OpenAI text-embedding-3  : 1536 dims, 8191 token limit, paid API
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIMS  = 768

print(f"Loading embedding model: {EMBEDDING_MODEL}")
print("(First run downloads ~420MB — subsequent runs are instant)")

model = SentenceTransformer(EMBEDDING_MODEL)

print(f"✓ Model loaded")


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: ChromaDB PersistentClient
#
# PersistentClient stores everything to disk at the given path.
# Every time your program restarts, the data is still there.
#
# The chroma_db/ folder structure:
#   chroma_db/
#     chroma.sqlite3       ← metadata, IDs, document text
#     <uuid>/              ← binary vector files (the actual embeddings)
#
# You never need to touch these files manually.
# To wipe everything and start fresh: delete the chroma_db/ folder.
#
# Alternative: chromadb.Client() — in-memory, wiped on restart.
# Use that only for unit tests.
# ─────────────────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="chroma_db")


def get_or_create_collection(collection_name: str = "documents"):
    """
    Gets an existing ChromaDB collection or creates it if it doesn't exist.

    🧠 CONCEPT: ChromaDB collections
    A collection is like a table in a relational DB — it groups related
    vectors together. We use one collection for all documents.
    Filtering by doc_id metadata lets us scope queries to one document
    when needed, without needing separate collections.

    hnsw:space = "cosine":
    Tells ChromaDB to use cosine similarity for nearest-neighbour search.
    Cosine measures the angle between vectors — not their length.
    This is standard for text because chunk length varies a lot.

    🧠 CONCEPT: HNSW index
    ChromaDB uses HNSW (Hierarchical Navigable Small World) internally.
    It's an approximate nearest neighbour algorithm — instead of comparing
    your query vector against every stored vector (slow), it navigates
    a graph structure to find approximate nearest neighbours in
    O(log n) time. Fast, accurate enough for RAG, industry standard.

    Alternatives: FAISS (Facebook), Annoy (Spotify), ScaNN (Google).
    ChromaDB uses HNSW so you don't have to think about this.
    """
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of text strings → list of 768-dim vectors.

    🧠 CONCEPT: batch encoding
    model.encode() accepts a list of strings and processes them together.
    This is much faster than encoding one at a time because:
      - The model processes inputs in parallel on CPU/GPU
      - Tokenization overhead is amortised across the batch

    show_progress_bar=True prints a tqdm progress bar during encoding.
    Useful when embedding hundreds of chunks.

    convert_to_list=True returns plain Python lists instead of numpy arrays.
    ChromaDB expects plain lists, not numpy arrays.

    normalize_embeddings=True:
    Divides each vector by its length so all vectors have magnitude = 1.
    WHY: When vectors are normalized, cosine similarity becomes equivalent
    to dot product — which is faster to compute. ChromaDB benefits from
    this when using cosine metric.
    Also recommended specifically for all-mpnet-base-v2 by its authors.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=False,        # return plain lists for ChromaDB
        normalize_embeddings=True,     # normalize for cosine similarity
        batch_size=32,                 # process 32 chunks at a time
    )
    return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]


def is_document_indexed(collection, doc_id: str) -> bool:
    """
    Checks if a document is already in ChromaDB.

    WHY THIS MATTERS:
    Without this check, running the pipeline twice on the same PDF
    creates duplicate chunks. Retrieval then returns the same chunk
    twice in results, wasting context window space and confusing the LLM.

    HOW IT WORKS:
    Query ChromaDB for any chunk with matching doc_id metadata.
    If even one exists → document is already indexed → skip.

    This is the core deduplication mechanism that JSON files couldn't
    give us. With ChromaDB it's a single metadata query.
    """
    results = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        limit=1,
    )
    return len(results["ids"]) > 0


def add_chunks_to_collection(collection, chunks: list[dict]) -> int:
    """
    Embeds chunks and stores them in ChromaDB.

    🧠 CONCEPT: ChromaDB's add() method
    Requires four parallel lists — all the same length, same order:

    ids        : unique string per chunk — used for deduplication and lookup
    embeddings : the vector for each chunk — what gets searched
    documents  : raw text — returned at retrieval time so LLM can read it
    metadatas  : filterable fields — page, doc_id, type, etc.

    WHY STORE DOCUMENTS (raw text) SEPARATELY FROM EMBEDDINGS:
    The embedding captures meaning but loses exact wording.
    At retrieval time the LLM needs the original text, not the vector.
    ChromaDB stores both and returns both on query.

    WHY METADATA IS SEPARATE FROM DOCUMENTS:
    Metadata fields are indexed for filtering.
    You can query: "find chunks similar to X, only from page 2"
    without that filter being part of the text the model searches.

    ChromaDB metadata limitation:
    Only primitive types: str, int, float, bool.
    No lists, no nested dicts. We flatten everything accordingly.

    Returns: number of chunks actually added
    """
    if not chunks:
        print("⚠ No chunks to add.")
        return 0

    doc_id = chunks[0]["doc_id"]

    # ── Deduplication ─────────────────────────────────────────────────────
    if is_document_indexed(collection, doc_id):
        print(f"⚠ Document '{doc_id}' already in ChromaDB — skipping.")
        print(f"  To re-index: call delete_document(collection, '{doc_id}') first")
        return 0

    print(f"\n🔢 Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    # ── Embed ─────────────────────────────────────────────────────────────
    texts      = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # ── Build parallel lists ──────────────────────────────────────────────
    ids       = [chunk["id"] for chunk in chunks]
    documents = texts

    # metadatas = [
    #     {
    #         "doc_id":      chunk["doc_id"],
    #         "source":      chunk["source"],
    #         "page":        chunk["page"],
    #         "type":        chunk["type"],
    #         "chunk_index": chunk["chunk_index"],
    #         "token_count": chunk["token_count"],
    #         "has_tables":  chunk["has_tables"],
    #     }
    #     for chunk in chunks
    # ]

    metadatas = [
        {
            "doc_id":      chunk["doc_id"],
            "source":      chunk["source"],
            "page":        chunk["page"],
            "type":        chunk["type"],
            "heading":     chunk.get("heading", ""),
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk.get("token_count", 0),
            "has_tables":  chunk.get("has_tables", False),
            "method":      chunk.get("method", "text"),
            "headers":     chunk.get("headers", ""),
            "accuracy":    float(chunk.get("accuracy", 0.0)),
        }
        for chunk in chunks
    ]

    # ── Store ─────────────────────────────────────────────────────────────
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")
    print(f"  Collection total: {collection.count()} chunks")

    return len(chunks)


def delete_document(collection, doc_id: str):
    """
    Removes all chunks for a document from ChromaDB.

    Use when:
    - You updated a PDF and want to re-index it
    - You want to remove a document from the system entirely

    The doc_id is stable (MD5 of filename) so re-running extraction
    on the same filename produces the same doc_id.
    """
    collection.delete(where={"doc_id": {"$eq": doc_id}})
    print(f"✓ Deleted all chunks for doc_id: {doc_id}")


def get_collection_stats(collection) -> dict:
    """
    Summary of what's currently stored in ChromaDB.
    Used by the pipeline runner to print a final report.
    """
    total = collection.count()

    if total == 0:
        return {"total_chunks": 0, "total_documents": 0, "sources": []}

    results = collection.get(include=["metadatas"])
    sources = list({m["source"] for m in results["metadatas"]})

    return {
        "total_chunks":    total,
        "total_documents": len(sources),
        "sources":         sources,
    }