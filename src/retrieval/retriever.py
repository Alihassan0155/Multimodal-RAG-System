from src.retrieval.graph import ChunkGraph
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDING_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """
    Full retrieval pipeline:
      BM25 keyword search  ─┐
                             ├── RRF fusion → Reranker → top-k chunks
      Vector search        ─┘
    """

    def __init__(self, top_candidates: int = 20):
        self.top_candidates = top_candidates

        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection    = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            raise ValueError("ChromaDB is empty. Run `python main.py ingest <pdf>` first.")

        print("  Loading corpus into BM25 index...")
        all_data         = self.collection.get(include=["documents", "metadatas"])
        self.chunk_ids   = all_data["ids"]
        self.chunk_texts = all_data["documents"]
        self.chunk_metas = all_data["metadatas"]

        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25        = BM25Okapi(tokenized_corpus)

        print(f"  Loading bi-encoder: {EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        all_chunks_for_graph = [
            {
                "id":          cid,
                "content":     self.chunk_texts[i],
                "source":      self.chunk_metas[i].get("source", ""),
                "page":        self.chunk_metas[i].get("page", 0),
                "chunk_index": self.chunk_metas[i].get("chunk_index", 0),
                "type":        self.chunk_metas[i].get("type", "text"),
            }
            for i, cid in enumerate(self.chunk_ids)
        ]
        self.graph = ChunkGraph(all_chunks_for_graph)

        print(f"  Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print(f"  ✓ Retriever ready — {len(self.chunk_ids)} chunks indexed\n")

    # ── Step 1: BM25 retrieval ────────────────────────────────────────────────
    def _bm25_retrieve(self, query: str) -> list[tuple[str, float]]:
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)
        top_indices     = np.argsort(scores)[::-1][: self.top_candidates]
        return [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 1e-10
        ]

    # ── Step 2: Vector retrieval ──────────────────────────────────────────────
    def _vector_retrieve(self, query: str) -> list[tuple[str, float]]:
        query_vector = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=False,
        )[0]

        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(self.top_candidates, self.collection.count()),
            include=["distances"],
        )

        ids          = results["ids"][0]
        distances    = results["distances"][0]
        similarities = [1 - d for d in distances]
        return list(zip(ids, similarities))

    # ── Step 3: RRF fusion ────────────────────────────────────────────────────
    def _rrf_fuse(
        self,
        bm25_results:   list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        rrf_scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Step 4: Reranker ──────────────────────────────────────────────────────
    def _rerank(self, query: str, candidate_ids: list[str], top_k: int) -> list[dict]:
        id_to_text = dict(zip(self.chunk_ids, self.chunk_texts))
        id_to_meta = dict(zip(self.chunk_ids, self.chunk_metas))

        valid_ids = [cid for cid in candidate_ids if cid in id_to_text]
        pairs     = [(query, id_to_text[cid]) for cid in valid_ids]
        scores    = self.reranker.predict(pairs, show_progress_bar=False)

        reranked = []
        for i, cid in enumerate(valid_ids):
            reranked.append({
                "id":           cid,
                "content":      id_to_text[cid],
                "rerank_score": float(scores[i]),
                **id_to_meta[cid],
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Threshold helpers ─────────────────────────────────────────────────────
    def _get_threshold(self, chunk: dict) -> float:
        chunk_type = chunk.get("type", "text")
        if chunk_type in ("table", "image"):
            return -8.0
        return -5.0

    # ── Metadata boosting ─────────────────────────────────────────────────────
    _STOPWORDS = {"what", "how", "is", "a", "an", "the", "are", "does", "do",
                  "in", "of", "to", "and", "or", "with", "for", "its", "it"}

    def _boost_metadata(self, chunks: list[dict], query: str) -> list[dict]:
        query_lower = query.lower()
        query_terms = set(query_lower.split()) - self._STOPWORDS

        for chunk in chunks:
            heading = chunk.get("heading", "")
            if not heading:
                first_line = chunk.get("content", "").split("\n")[0].strip()
                if len(first_line) < 100:
                    heading = first_line

            heading_terms = set(heading.lower().split()) - self._STOPWORDS
            overlap = len(query_terms & heading_terms)
            boost = 2.0 if overlap >= 3 else 1.0 if overlap >= 2 else 0.5 if overlap >= 1 else 0.0

            chunk["rerank_score"] = chunk.get("rerank_score", 0.0) + boost

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks

    # ── Diversity selection ────────────────────────────────────────────────────
    def _diversify(self, chunks: list[dict], top_k: int) -> list[dict]:
        """
        Picks top_k chunks while capping at 2 per page.
        Prevents the result from being 5 chunks from the same dense page
        while ignoring the page that actually defines the concept.
        Overflow chunks fill remaining slots so we always return top_k.
        """
        selected: list[dict] = []
        page_counts: dict[int, int] = {}
        overflow: list[dict] = []

        for chunk in chunks:
            page = chunk.get("page", 0)
            if page_counts.get(page, 0) < 2:
                selected.append(chunk)
                page_counts[page] = page_counts.get(page, 0) + 1
            else:
                overflow.append(chunk)
            if len(selected) >= top_k:
                break

        for chunk in overflow:
            if len(selected) >= top_k:
                break
            selected.append(chunk)

        return selected[:top_k]

    # ── Public: single query ──────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.retrieve_multi([query], original_query=query, top_k=top_k)

    # ── Public: multi-query hybrid retrieval ──────────────────────────────────
    def retrieve_multi(
        self,
        queries:          list[str],
        original_query:   str,
        top_k:            int   = 5,
        intent_threshold: float = None,
    ) -> list[dict]:
        """
        Runs BM25 + vector for each query variant, fuses via global RRF,
        expands with graph neighbours, reranks against the original query,
        then filters by per-chunk relevance threshold.
        """
        all_bm25_results:   dict[str, list] = {}
        all_vector_results: dict[str, list] = {}

        for i, q in enumerate(queries):
            label = "original" if i == 0 else f"variant_{i}"
            print(f"  [BM25]   {label}: '{q[:50]}'")
            all_bm25_results[label] = self._bm25_retrieve(q)
            print(f"  [Vector] {label}: '{q[:50]}'")
            all_vector_results[label] = self._vector_retrieve(q)

        print(f"  [RRF]    fusing {len(queries) * 2} ranked lists...")
        global_rrf: dict[str, float] = {}
        k = 60

        for results in all_bm25_results.values():
            for rank, (chunk_id, _) in enumerate(results, start=1):
                global_rrf[chunk_id] = global_rrf.get(chunk_id, 0.0) + 1 / (k + rank)

        for results in all_vector_results.values():
            for rank, (chunk_id, _) in enumerate(results, start=1):
                global_rrf[chunk_id] = global_rrf.get(chunk_id, 0.0) + 1 / (k + rank)

        fused         = sorted(global_rrf.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [cid for cid, _ in fused[: self.top_candidates]]
        print(f"  [RRF]    {len(candidate_ids)} unique candidates after fusion")

        # Graph expansion
        id_to_chunk = {
            cid: {
                "id":          cid,
                "content":     self.chunk_texts[i],
                "source":      self.chunk_metas[i].get("source", ""),
                "page":        self.chunk_metas[i].get("page", 0),
                "chunk_index": self.chunk_metas[i].get("chunk_index", 0),
                "type":        self.chunk_metas[i].get("type", "text"),
            }
            for i, cid in enumerate(self.chunk_ids)
        }

        candidate_chunks = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
        expanded_chunks  = self.graph.expand(candidate_chunks, top_n=2) or candidate_chunks
        candidate_ids    = [c["id"] for c in expanded_chunks]

        # Reranker over 2× top_k so boosting + diversity have room to reorder
        print(f"  [Rerank] scoring against original query...")
        reranked = self._rerank(original_query, candidate_ids, top_k * 2)

        for c in reranked[:3]:
            print(f"  DEBUG: id={c['id']} type={c.get('type','MISSING')} score={c.get('rerank_score',0):.2f}")

        # Metadata boost — heading-aware score adjustment
        reranked = self._boost_metadata(reranked, original_query)

        # Relevance threshold filter
        relevant = []
        for c in reranked:
            base = min(
                self._get_threshold(c),
                intent_threshold if intent_threshold is not None else -5.0,
            )
            effective = base - 2 if c.get("type") == "table" else base
            if c["rerank_score"] >= effective:
                relevant.append(c)

        if not relevant:
            print("  [Threshold] No chunks passed — falling back to all reranked")
            relevant = reranked

        print(f"  [Threshold] {len(relevant)} chunks passed")

        # Diversity selection — max 2 chunks per page, then fill to top_k
        final = self._diversify(relevant, top_k)
        print(f"  [Diversity] {len(final)} chunks selected (top_k={top_k})")
        return final
