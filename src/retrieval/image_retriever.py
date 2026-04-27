"""
CLIP-based image retrieval.

Embeddings are stored as a numpy array on disk alongside a JSON metadata file.
No extra vector DB needed — cosine similarity over ≤ a few hundred images
is negligible work for numpy.

Scoring:
    final_score = 0.7 * clip_similarity + 0.3 * text_context_overlap
"""

import json
from pathlib import Path

import numpy as np

ROOT            = Path(__file__).parent.parent.parent
INDEX_DIR       = ROOT / "data" / "image_index"
METADATA_FILE   = INDEX_DIR / "metadata.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"

CLIP_MODEL      = "clip-ViT-B-32"
SIM_THRESHOLD   = 0.17


class ImageRetriever:
    def __init__(self):
        self._model     = None
        self._meta: list[dict]          = []
        self._embs: np.ndarray | None   = None
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if METADATA_FILE.exists():
            self._meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        if EMBEDDINGS_FILE.exists() and self._meta:
            self._embs = np.load(str(EMBEDDINGS_FILE)).astype(np.float32)

    def _save(self):
        METADATA_FILE.write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self._embs is not None and len(self._embs):
            np.save(str(EMBEDDINGS_FILE), self._embs)
        elif EMBEDDINGS_FILE.exists():
            EMBEDDINGS_FILE.unlink()

    # ── CLIP model (lazy load) ────────────────────────────────────────────────

    def _clip(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  [ImageRetriever] Loading {CLIP_MODEL}...")
            self._model = SentenceTransformer(CLIP_MODEL)
        return self._model

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_images(self, images: list[dict]) -> None:
        """Embed and store images. Re-ingesting the same doc_id replaces old entries."""
        if not images:
            return

        doc_id = images[0]["doc_id"]
        self._remove_doc(doc_id)

        from PIL import Image as PILImage

        model       = self._clip()
        new_meta    = []
        new_vectors = []

        for img in images:
            path = Path(img["image_path"])
            if not path.exists():
                continue
            try:
                pil_img = PILImage.open(path).convert("RGB")

                img_vec = model.encode(pil_img, normalize_embeddings=True)

                text_ctx = " ".join(filter(None, [
                    img.get("caption", ""),
                    img.get("context", "")[:300],
                ]))
                if text_ctx.strip():
                    txt_vec = model.encode(text_ctx, normalize_embeddings=True)
                    combined = 0.7 * img_vec + 0.3 * txt_vec
                    norm = np.linalg.norm(combined)
                    combined = combined / norm if norm > 0 else img_vec
                else:
                    combined = img_vec

                new_meta.append(img)
                new_vectors.append(combined.astype(np.float32))

            except Exception as e:
                print(f"  [ImageRetriever] Skipped {img.get('image_id')}: {e}")

        if not new_vectors:
            return

        new_arr = np.array(new_vectors, dtype=np.float32)
        if self._embs is None or len(self._meta) == 0:
            self._embs = new_arr
        else:
            self._embs = np.vstack([self._embs, new_arr])

        self._meta.extend(new_meta)
        self._save()
        print(f"  [ImageRetriever] Indexed {len(new_vectors)} image(s). "
              f"Total: {len(self._meta)}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return images above SIM_THRESHOLD, ranked by final_score.

        top_k=None (default) returns every candidate that passes the base
        threshold.  The caller owns the selection policy (how many to show
        and what intent-specific threshold to enforce).
        """
        if self._embs is None or len(self._meta) == 0:
            return []

        model = self._clip()
        q_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)

        clip_scores = (self._embs @ q_vec).tolist()
        query_terms = set(query.lower().split())
        results     = []

        for idx, clip_sim in enumerate(clip_scores):
            if clip_sim < SIM_THRESHOLD:
                continue
            meta      = self._meta[idx]
            ctx       = (meta.get("context", "") + " " + meta.get("caption", "")).lower()
            ctx_words = set(ctx.split())

            overlap = len(query_terms & ctx_words) / max(len(query_terms), 1)
            final   = 0.7 * clip_sim + 0.3 * overlap
            results.append({**meta, "clip_score": round(clip_sim, 4),
                             "similarity_score": round(final, 4)})

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results if top_k is None else results[:top_k]

    # ── Deletion ──────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> None:
        self._remove_doc(doc_id)
        self._save()

    def _remove_doc(self, doc_id: str) -> None:
        if not self._meta:
            return
        paired = list(zip(self._meta, self._embs if self._embs is not None else [None] * len(self._meta)))
        kept   = [(m, e) for m, e in paired if m.get("doc_id") != doc_id]
        if kept:
            metas, embs = zip(*kept)
            self._meta = list(metas)
            self._embs = np.array(embs, dtype=np.float32)
        else:
            self._meta = []
            self._embs = None
