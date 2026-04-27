"""
FastAPI server for the Multimodal RAG system.

Fixes applied:
  1. load_dotenv() called immediately — Groq API key available to all modules
  2. BASE_DIR / absolute paths — ChromaDB found regardless of cwd
  3. Models and retriever loaded ONCE at startup — not per-request
  4. Sync ML calls wrapped properly — no event-loop blocking
  5. Detailed startup checks — prints exactly what's missing on boot
"""

import os
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

# ── Load .env FIRST — before any src imports ─────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Absolute paths — safe regardless of working directory ────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
CHROMA_PATH = str(BASE_DIR / "chroma_db")
UPLOAD_DIR  = BASE_DIR / "data" / "uploads"
IMAGES_DIR  = BASE_DIR / "data" / "extracted" / "images"

# ── Module-level singletons (loaded once at startup) ─────────────────────────
_retriever       = None
_image_retriever = None


def _load_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever(top_candidates=20)
    return _retriever


def _load_image_retriever():
    global _image_retriever
    if _image_retriever is None:
        from src.retrieval.image_retriever import ImageRetriever
        _image_retriever = ImageRetriever()
    return _image_retriever


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — validate environment and pre-load models at startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("  Multimodal RAG API — startup checks")
    print("="*60)

    # 1. Check Groq API key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("  [WARN] GROQ_API_KEY is not set — /api/query will fail!")
        print("         Add it to your .env file: GROQ_API_KEY=gsk_...")
    else:
        print(f"  [OK]  GROQ_API_KEY found ({groq_key[:8]}...)")

    # 2. Check ChromaDB path
    chroma_dir = Path(CHROMA_PATH)
    if chroma_dir.exists():
        print(f"  [OK]  ChromaDB found at: {CHROMA_PATH}")
    else:
        print(f"  [INFO] ChromaDB not found yet — will be created on first ingest")
        print(f"         Path: {CHROMA_PATH}")

    # 3. Pre-load text retriever (SentenceTransformer + cross-encoder)
    print("  [..] Loading retriever models (this may take 30-60s first time)...")
    try:
        await run_in_threadpool(_load_retriever)
        print("  [OK]  Retriever ready")
    except Exception as e:
        print(f"  [WARN] Retriever failed to pre-load: {e}")
        traceback.print_exc()

    # 4. Pre-load vectorstore embedding model (avoids HF Hub network call in ingest threads)
    print("  [..] Loading vectorstore embedding model...")
    try:
        await run_in_threadpool(lambda: __import__("src.pipeline.vectorstore", fromlist=["model"]))
        print("  [OK]  Vectorstore embedding model ready")
    except Exception as e:
        print(f"  [WARN] Vectorstore model failed to pre-load: {e}")

    # 5. Pre-load image retriever (CLIP — loads on first image query otherwise)
    print("  [..] Loading image retriever (CLIP)...")
    try:
        await run_in_threadpool(_load_image_retriever)
        print("  [OK]  Image retriever ready")
    except Exception as e:
        print(f"  [WARN] Image retriever failed to pre-load: {e}")

    print("="*60)
    print(f"  API docs  →  http://localhost:8000/docs")
    print(f"  Frontend  →  http://localhost:5173")
    print("="*60 + "\n")

    yield

    print("\n[*] Shutting down RAG API...")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG API",
    description="REST API for PDF ingestion, retrieval, and RAG-based Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


class QueryResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    sources: list[dict]
    expanded_queries: list[str]
    images: list[dict] = []


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    documents: list[dict]


class IngestResponse(BaseModel):
    status: str
    filename: str
    message: str


class DeleteResponse(BaseModel):
    status: str
    doc_id: str
    remaining_chunks: int


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — thin wrappers so async endpoints can call sync ML code safely
# ─────────────────────────────────────────────────────────────────────────────

# Words that signal an explicit request for a visual output.
_STRONG_VISUAL = {
    "show", "diagram", "figure", "picture", "visual", "visualize",
    "visualise", "illustrate", "illustration", "display", "draw",
    "plot", "depict", "sketch", "schematic", "chart", "graph",
}

# Words where a relevant image would genuinely complement the answer.
_MEDIUM_VISUAL = {"architecture", "structure", "design", "layout", "flow", "overview"}

# Minimum similarity_score to return an image for each intent tier.
_STRONG_THRESHOLD = 0.25   # lower bar — user explicitly asked for a visual
_MEDIUM_THRESHOLD = 0.35   # higher bar — image must be highly relevant to be shown


def _visual_intent(query: str) -> str:
    """Return 'strong', 'medium', or 'none' based on query keywords."""
    q = query.lower()
    if any(k in q for k in _STRONG_VISUAL):
        return "strong"
    if any(k in q for k in _MEDIUM_VISUAL):
        return "medium"
    return "none"


def _run_ingest(file_path: str):
    """Blocking ingest — runs in thread pool."""
    from src.pipeline.ingest_pipeline import ingest
    ingest(file_path)


def _run_query(question: str, history: list[dict] = None):
    """
    Blocking full RAG pipeline — runs in thread pool.
    Returns a dict that maps directly to QueryResponse fields.
    """
    from src.pipeline.query_classifier import classify_query
    from src.pipeline.answerer import expand_query, answer

    # Step 0: Intent classification
    intent_config   = classify_query(question)
    intent          = intent_config["intent"]
    top_k           = intent_config["top_k"]
    expand_enabled  = intent_config["expand_queries"]
    threshold       = intent_config["threshold"]

    # Step 1: Query expansion
    expanded_queries = expand_query(question) if expand_enabled else [question]

    # Step 2: Retrieval — reuse the singleton retriever
    retriever = _load_retriever()
    chunks = retriever.retrieve_multi(
        queries=expanded_queries,
        original_query=question,
        top_k=top_k,
        intent_threshold=threshold,
    )

    # Step 3: Image retrieval — precision-first, relevance-driven.
    #
    # strong intent → return the single best candidate above _STRONG_THRESHOLD
    # medium intent → return the single best candidate only if above _MEDIUM_THRESHOLD
    # none          → skip entirely
    images = []
    v_intent = _visual_intent(question)
    if v_intent != "none":
        try:
            img_retriever = _load_image_retriever()
            candidates    = img_retriever.retrieve(question)   # all above base threshold
            if candidates:
                best    = candidates[0]                        # highest similarity_score
                cutoff  = _STRONG_THRESHOLD if v_intent == "strong" else _MEDIUM_THRESHOLD
                if best["similarity_score"] >= cutoff:
                    images = [{
                        "image_id": best["image_id"],
                        "url":      f"/images/{best['image_filename']}",
                        "source":   best["source"],
                        "page":     best["page"],
                        "caption":  best.get("caption", ""),
                        "score":    best["similarity_score"],
                    }]
        except Exception as e:
            print(f"  [Images] Retrieval skipped — {e}")

    # Step 4: Generate answer
    response_text = answer(
        query=question,
        chunks=chunks,
        intent=intent,
        history=history,
    )

    sources = [
        {
            "source":          c.get("source", "unknown"),
            "page":            c.get("page", "?"),
            "type":            c.get("type", "text"),
            "rerank_score":    round(c.get("rerank_score", 0), 3),
            "content_preview": c.get("content", "")[:200] + "...",
        }
        for c in chunks
    ]

    return {
        "answer":           response_text,
        "intent":           intent,
        "confidence":       intent_config.get("confidence", 0.0),
        "sources":          sources,
        "expanded_queries": expanded_queries,
        "images":           images,
    }


def _get_chroma_collection():
    """Return the ChromaDB collection using the absolute path."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection("documents")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest(file: UploadFile = File(...)):
    """Upload a PDF and run the full ingestion pipeline."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    try:
        # Save file
        content = await file.read()
        file_path.write_bytes(content)

        # Run blocking pipeline in thread pool — won't block the event loop
        await run_in_threadpool(_run_ingest, str(file_path))

        # Reload the image retriever's in-memory index from disk so the new
        # images are visible to queries without restarting the server.
        # (_load_image_retriever keeps the CLIP model warm; _load refreshes data.)
        try:
            ir = _load_image_retriever()
            await run_in_threadpool(ir._load)
        except Exception:
            pass

        return IngestResponse(
            status="success",
            filename=file.filename,
            message=f"Successfully ingested {file.filename}",
        )

    except Exception as e:
        traceback.print_exc()   # Full traceback in your terminal — read this!
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    """Return knowledge base statistics."""

    def _stats():
        collection = _get_chroma_collection()
        total = collection.count()
        if total == 0:
            return {"total_chunks": 0, "total_documents": 0, "documents": []}

        results = collection.get(include=["metadatas"])
        docs: dict = {}
        for m in results["metadatas"]:
            did = m.get("doc_id", "unknown")
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "source": m.get("source", "unknown"),
                    "chunks": 0,
                    "pages":  set(),
                    "types":  set(),
                }
            docs[did]["chunks"] += 1
            docs[did]["pages"].add(m.get("page", 0))
            docs[did]["types"].add(m.get("type", "text"))

        return {
            "total_chunks":    total,
            "total_documents": len(docs),
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "source": d["source"],
                    "chunks": d["chunks"],
                    "pages":  sorted(d["pages"]),
                    "types":  sorted(d["types"]),
                }
                for d in docs.values()
            ],
        }

    try:
        result = await run_in_threadpool(_stats)
        return StatsResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to read stats: {str(e)}")


@app.delete("/api/docs/{doc_id}", response_model=DeleteResponse)
async def api_delete(doc_id: str):
    """Delete all chunks for a document by its doc_id."""

    def _delete():
        collection = _get_chroma_collection()
        results = collection.get(where={"doc_id": {"$eq": doc_id}}, limit=1)
        if not results["ids"]:
            raise ValueError(f"No document found with doc_id '{doc_id}'")
        collection.delete(where={"doc_id": {"$eq": doc_id}})
        try:
            _load_image_retriever().delete_document(doc_id)
        except Exception as e:
            print(f"  [Images] Delete skipped — {e}")
        return collection.count()

    try:
        remaining = await run_in_threadpool(_delete)
        return DeleteResponse(status="success", doc_id=doc_id, remaining_chunks=remaining)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.post("/api/reindex-images")
async def api_reindex_images():
    """
    Re-extract and re-index images for every PDF already in the knowledge base.
    Use this once after upgrading to the multimodal pipeline on an existing install.
    """
    def _reindex():
        from src.ingest.extract_images import extract_images_from_pdf
        from src.retrieval.image_retriever import ImageRetriever

        collection = _get_chroma_collection()
        if collection.count() == 0:
            return {"indexed": 0, "message": "Knowledge base is empty — ingest a PDF first."}

        # Collect unique (doc_id, source_filename) pairs from ChromaDB
        results = collection.get(include=["metadatas"])
        seen: dict[str, str] = {}          # doc_id → source filename
        for m in results["metadatas"]:
            did = m.get("doc_id")
            src = m.get("source")
            if did and src and did not in seen:
                seen[did] = src

        img_retriever = ImageRetriever()
        total_images  = 0
        processed     = []

        for doc_id, source in seen.items():
            # Try to locate the original PDF in uploads or pdfs directories
            pdf_path = None
            for search_dir in [UPLOAD_DIR, BASE_DIR / "data" / "pdfs"]:
                candidate = search_dir / source
                if candidate.exists():
                    pdf_path = candidate
                    break

            if pdf_path is None:
                print(f"  [Reindex] PDF not found for '{source}' — skipping")
                continue

            try:
                images = extract_images_from_pdf(str(pdf_path), doc_id)
                if images:
                    img_retriever.index_images(images)
                    total_images += len(images)
                processed.append({"doc_id": doc_id, "source": source, "images": len(images)})
            except Exception as e:
                print(f"  [Reindex] Failed for {source}: {e}")

        return {"indexed": total_images, "documents": processed}

    try:
        result = await run_in_threadpool(_reindex)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """Run the full RAG pipeline and return a structured answer."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert Pydantic history models to plain dicts for the pipeline
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        result = await run_in_threadpool(_run_query, question, history)
        return QueryResponse(**result)
    except Exception as e:
        traceback.print_exc()   # ALWAYS read the terminal — this tells you the real error
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Health check — useful for debugging ──────────────────────────────────────
@app.get("/api/health")
async def health():
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    chroma_ok = Path(CHROMA_PATH).exists()
    return {
        "status":       "ok",
        "groq_api_key": "set" if groq_ok else "MISSING — queries will fail",
        "chroma_db":    CHROMA_PATH,
        "chroma_exists": chroma_ok,
        "upload_dir":   str(UPLOAD_DIR),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)