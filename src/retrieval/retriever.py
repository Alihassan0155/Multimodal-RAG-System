# src/retrieval/retriever.py
from src.retrieval.graph import ChunkGraph
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: Two types of models working together
#
# 1. BI-ENCODER (SentenceTransformer — same model as ingest)
#    Encodes query and chunks SEPARATELY into vectors.
#    Fast — chunk vectors are pre-computed and stored in ChromaDB.
#    At query time: only the query needs encoding (milliseconds).
#    Quality: good, but can miss nuance since texts are encoded in isolation.
#
# 2. CROSS-ENCODER (CrossEncoder — the reranker)
#    Reads query AND chunk TOGETHER in one forward pass.
#    Sees the full context of both → much more accurate relevance scoring.
#    Slow — must process every (query, chunk) pair fresh at query time.
#    This is why we only run it on the top 20 candidates, not all chunks.
#
# The hybrid pipeline exploits both:
#   Bi-encoder speed for broad retrieval → Cross-encoder quality for final ranking
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """
    Full retrieval pipeline:
      BM25 keyword search  ─┐
                             ├── RRF fusion → Reranker → top-k chunks
      Vector search        ─┘

    Usage:
        retriever = HybridRetriever()
        chunks = retriever.retrieve("what is antibiotic resistance?", top_k=5)
    """

    def __init__(self, top_candidates: int = 20):
        """
        top_candidates: how many chunks each retriever fetches before RRF.
        These are the candidates that go into the reranker.
        Keep this at 20 — enough diversity for RRF, small enough for fast reranking.
        """
        self.top_candidates = top_candidates

        # ── Load ChromaDB ─────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection    = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            raise ValueError("ChromaDB is empty. Run `python main.py ingest <pdf>` first.")

        # ── Load all chunks from ChromaDB into memory for BM25 ───────────
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: why BM25 needs all chunks in memory
        #
        # BM25 is not a vector search — it's a term-frequency algorithm.
        # It needs to know the full corpus to compute IDF (inverse document
        # frequency). IDF asks: "how rare is this word across ALL chunks?"
        # A word rare in the corpus gets a high IDF score → high signal.
        # A word common everywhere gets low IDF → low signal.
        #
        # This means BM25 must see ALL chunks at init time — you can't
        # query it chunk by chunk like a vector DB.
        #
        # For large corpora (millions of chunks) this would be a RAM problem.
        # For our use case (hundreds to low thousands of chunks) it's fine.
        #
        # Alternatives for large scale: Elasticsearch, OpenSearch, Typesense
        # — these run BM25 as a server so the corpus lives on disk.
        # ─────────────────────────────────────────────────────────────────
        print("  Loading corpus into BM25 index...")
        all_data   = self.collection.get(include=["documents", "metadatas"])
        self.chunk_ids   = all_data["ids"]
        self.chunk_texts = all_data["documents"]
        self.chunk_metas = all_data["metadatas"]

        # Tokenize corpus for BM25
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: BM25 tokenization
        #
        # BM25Okapi expects a list of token lists — one per document.
        # We do simple lowercase + whitespace split here.
        #
        # Better alternatives:
        #   - nltk.word_tokenize() : handles punctuation properly
        #   - spacy tokenizer      : linguistic awareness
        #
        # For RAG with technical text, simple splitting works well enough.
        # The bigger gains come from the reranker, not tokenizer quality.
        # ─────────────────────────────────────────────────────────────────
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25        = BM25Okapi(tokenized_corpus)

        # ── Load bi-encoder (same model used at ingest) ───────────────────
        print(f"  Loading bi-encoder: {EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # ── Build chunk graph ─────────────────────────────────────────────────
        # Build in-memory graph of chunk relationships.
        # Used at query time to expand retrieved chunks with neighbours.
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

        # ── Load cross-encoder reranker ───────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: CrossEncoder
        #
        # CrossEncoder("model-name") loads a model fine-tuned specifically
        # for passage re-ranking. It was trained on millions of
        # (query, passage, relevance_label) triplets.
        #
        # .predict([(query, text), ...]) returns a raw score per pair.
        # Scores are NOT probabilities — just comparable within a batch.
        # Higher score = more relevant to the query.
        #
        # ms-marco-MiniLM-L-6-v2:
        #   - Trained on MS MARCO (Microsoft Machine Reading Comprehension)
        #   - 6-layer MiniLM base → fast on CPU
        #   - ~80MB, loads in ~2 seconds
        # ─────────────────────────────────────────────────────────────────
        print(f"  Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print(f"  ✓ Retriever ready — {len(self.chunk_ids)} chunks indexed\n")

    # ── Step 1: BM25 retrieval ────────────────────────────────────────────────
    def _bm25_retrieve(self, query: str) -> list[tuple[str, float]]:
        """
        Keyword search over all chunks using BM25Okapi.

        Returns: [(chunk_id, bm25_score), ...] sorted by score descending.

        🧠 CONCEPT: BM25Okapi formula
        For each chunk, score = Σ IDF(term) × TF-saturation(term, chunk)

        TF saturation: unlike raw TF, BM25 caps term frequency contribution.
        Mentioning a word 10 times vs 5 times doesn't double the score —
        it has diminishing returns. This prevents spam-like repetition
        from gaming the ranking.

        IDF: log((N - df + 0.5) / (df + 0.5))
        Where N = total chunks, df = chunks containing this term.
        Rare terms get high IDF. Common terms ("the", "is") get near-zero.
        """
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)

        # Get indices of top candidates sorted by score descending
        top_indices = np.argsort(scores)[::-1][: self.top_candidates]

        return [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 1e-10   # ignore chunks with zero BM25 score
        ]

    # ── Step 2: Vector retrieval ──────────────────────────────────────────────
    def _vector_retrieve(self, query: str) -> list[tuple[str, float]]:
        """
        Dense semantic search via ChromaDB.

        Returns: [(chunk_id, similarity_score), ...] sorted descending.

        🧠 CONCEPT: why embed the query with the same model as chunks
        Vector search works by finding chunk vectors closest (by cosine
        similarity) to the query vector. This only makes sense if both
        live in the same vector space — i.e. produced by the same model.
        Using a different model at query time is like asking for directions
        in French to someone who only understands Mandarin.
        """
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

        ids       = results["ids"][0]
        distances = results["distances"][0]

        # Convert distance → similarity
        # cosine distance = 1 - cosine_similarity
        # so similarity   = 1 - distance
        similarities = [1 - d for d in distances]

        return list(zip(ids, similarities))

    # ── Step 3: RRF fusion ────────────────────────────────────────────────────
    def _rrf_fuse(
        self,
        bm25_results:   list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion — merges two ranked lists into one.

        🧠 CONCEPT: RRF formula
        For each chunk that appears in either list:
            RRF_score = Σ  1 / (k + rank)

        k=60 is from the original RRF paper (Cormack et al. 2009).
        It was found empirically across many IR benchmarks.
        It dampens the effect of rank 1 vs rank 2 — prevents one system
        from completely dominating just because it's slightly more confident.

        WHY RRF INSTEAD OF SCORE AVERAGING:
        BM25 scores are unbounded (could be 14.2 or 0.3).
        Cosine similarities are bounded (0.0 to 1.0).
        You cannot average these — they live in different numeric spaces.
        RRF ignores the raw scores entirely and only uses RANK POSITION.
        This makes it scale-invariant — works regardless of score ranges.

        A chunk ranked #1 by both systems scores:
            1/(60+1) + 1/(60+1) = 0.0328
        A chunk ranked #1 by one system and #5 by the other:
            1/(60+1) + 1/(60+5) = 0.0163 + 0.0154 = 0.0317
        Both are high — consensus between systems is rewarded.

        WHERE ELSE IS RRF USED:
        Microsoft Cognitive Search, Elastic hybrid search,
        most production RAG systems at scale.
        """
        rrf_scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

        # Sort by RRF score descending
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Step 4: Reranker ──────────────────────────────────────────────────────
    def _rerank(self, query: str, candidate_ids: list[str], top_k: int) -> list[dict]:
        """
        Cross-encoder reranking on the RRF candidates.

        🧠 CONCEPT: why reranking improves over RRF alone
        RRF fuses rankings — it rewards chunks that appear high in both
        BM25 and vector results. But it still can't read the query and
        chunk together. It has no idea if the chunk actually ANSWERS
        the question — just that it scored well in retrieval.

        The cross-encoder reads (query + chunk) jointly. It can detect:
          - "This chunk mentions the keywords but doesn't answer the question"
          - "This chunk is semantically close but answers a different question"
          - "This chunk perfectly answers what was asked"

        This is why rerankers consistently improve RAG quality by 10-30%
        in benchmarks, even when the retrieval step is already good.

        We run it only on top_candidates (20) not all chunks because
        CrossEncoder.predict() is O(n) — running on 10,000 chunks per
        query would take minutes. On 20 candidates: milliseconds.
        """
        # Build a lookup: id → {text, metadata}
        id_to_text = dict(zip(self.chunk_ids, self.chunk_texts))
        id_to_meta = dict(zip(self.chunk_ids, self.chunk_metas))

        # Filter to candidates that exist in our corpus
        valid_ids = [cid for cid in candidate_ids if cid in id_to_text]

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, id_to_text[cid]) for cid in valid_ids]

        # Score all pairs
        # ─────────────────────────────────────────────────────────────────
        # 🧠 CONCEPT: CrossEncoder.predict() output
        # Returns a numpy array of raw logit scores — one per pair.
        # These are NOT probabilities (no softmax applied).
        # They ARE comparable within this batch:
        #   higher score → more relevant to this specific query.
        # show_progress_bar=False keeps output clean.
        # ─────────────────────────────────────────────────────────────────
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        reranked = []
        for i, cid in enumerate(valid_ids):
            reranked.append({
                "id":           cid,
                "content":      id_to_text[cid],
                "rerank_score": float(scores[i]),
                **id_to_meta[cid],   # page, source, doc_id, type, etc.
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Public method ─────────────────────────────────────────────────────────
    # def retrieve(self, query: str, top_k: int = 5) -> list[dict]:

        # """
        # Full pipeline: BM25 → Vector → RRF → Rerank → top_k chunks.

        # Args:
        #     query  : natural language question
        #     top_k  : number of final chunks to return (goes to LLM)

        # Returns:
        #     List of chunk dicts, best match first. Each has:
        #     id, content, rerank_score, source, page, doc_id, type, ...
        # """
        # print(f"  [BM25]   searching...")
        # bm25_results   = self._bm25_retrieve(query)
        # print(f"  [BM25]   {len(bm25_results)} candidates")

        # print(f"  [Vector] searching...")
        # vector_results = self._vector_retrieve(query)
        # print(f"  [Vector] {len(vector_results)} candidates")

        # print(f"  [RRF]    fusing...")
        # fused          = self._rrf_fuse(bm25_results, vector_results)
        # candidate_ids  = [cid for cid, _ in fused[: self.top_candidates]]
        # print(f"  [RRF]    {len(candidate_ids)} merged candidates")

        # print(f"  [Rerank] scoring {len(candidate_ids)} candidates...")
        # final_chunks   = self._rerank(query, candidate_ids, top_k)
        # print(f"  [Rerank] top {len(final_chunks)} selected\n")

        # MIN_RERANK_SCORE = -5.0

        # relevant_chunks = [c for c in final_chunks if c["rerank_score"] >= MIN_RERANK_SCORE]

        # if not relevant_chunks:
        #  print("  [Threshold] No chunks passed relevance threshold — query likely out of domain")
        #  return []

        # print(f"  [Threshold] {len(relevant_chunks)}/{len(final_chunks)} chunks passed (score >= {MIN_RERANK_SCORE})")
        # return 
    # ── Type-aware + intent-aware threshold ──────────────────────────────
    def effective_threshold(chunk: dict) -> float:
        # Intent threshold overrides type threshold if more lenient
        type_thresh   = self._get_threshold(chunk)
        intent_thresh = intent_threshold if intent_threshold is not None else -5.0
        # Use whichever is more lenient (lower number)
        return min(type_thresh, intent_thresh)
        
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Full pipeline with query expansion support.
        Accepts original query — expansion happens in answerer.py.
        This method also accepts a list of queries for multi-query retrieval.
        """
        return self.retrieve_multi([query], original_query=query, top_k=top_k)


    def retrieve_multi(self, queries: list[str], original_query: str, top_k: int = 5,intent_threshold: float = None) -> list[dict]:
        """
        Multi-query hybrid retrieval.

        Runs BM25 + vector for EACH query variant. Merges all results, deduplicates, applies global RRF.Reranks against the ORIGINAL query only.
        """
    
        
        
        all_bm25_results:   dict[str, list[tuple[str, float]]] = {}
        all_vector_results: dict[str, list[tuple[str, float]]] = {}

        # Run retrieval for each query variant
        for i, q in enumerate(queries):
            label = "original" if i == 0 else f"variant_{i}"
            print(f"  [BM25]   {label}: '{q[:50]}'")
            all_bm25_results[label]   = self._bm25_retrieve(q)

            print(f"  [Vector] {label}: '{q[:50]}'")
            all_vector_results[label] = self._vector_retrieve(q)

         # Global RRF across ALL results from ALL queries
        print(f"  [RRF]    fusing {len(queries) * 2} ranked lists...")
        global_rrf: dict[str, float] = {}
        k = 60

        for label, results in all_bm25_results.items():
            for rank, (chunk_id, _) in enumerate(results, start=1):
                global_rrf[chunk_id] = global_rrf.get(chunk_id, 0.0) + 1 / (k + rank)

          
        for label, results in all_vector_results.items():
            for rank, (chunk_id, _) in enumerate(results, start=1):
                global_rrf[chunk_id] = global_rrf.get(chunk_id, 0.0) + 1 / (k + rank)

        # Sort by global RRF score
        fused         = sorted(global_rrf.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [cid for cid, _ in fused[: self.top_candidates]]

        print(f"  [RRF]    {len(candidate_ids)} unique candidates after dedup + fusion")

       # ── Graph expansion ───────────────────────────────────────────────────
        # Build chunk lookup from in-memory corpus
        id_to_chunk = {}
        for i, cid in enumerate(self.chunk_ids):
            id_to_chunk[cid] = {
            "id":          cid,
            "content":     self.chunk_texts[i],
            "source":      self.chunk_metas[i].get("source", ""),
            "page":        self.chunk_metas[i].get("page", 0),
            "chunk_index": self.chunk_metas[i].get("chunk_index", 0),
            "type":        self.chunk_metas[i].get("type", "text"),
        }

        candidate_chunks = [
            id_to_chunk[cid]
            for cid in candidate_ids
            if cid in id_to_chunk
        ]

        expanded_chunks = self.graph.expand(candidate_chunks, top_n=2)
        if expanded_chunks is None:
            expanded_chunks = candidate_chunks

        candidate_ids = [c["id"] for c in expanded_chunks]

        # Rerank against ORIGINAL query
        print(f"  [Rerank] scoring against original query...")
        final_chunks = self._rerank(original_query, candidate_ids, top_k)

        # Relevance threshold
        relevant = [
            c for c in final_chunks
            if c["rerank_score"] >= min(
            self._get_threshold(c),
            intent_threshold if intent_threshold is not None else -5.0
            )
        ]

        if not relevant:
            print("  [Threshold] No chunks passed — query likely out of domain")
            return []

        print(f"  [Threshold] {len(relevant)}/{len(final_chunks)} chunks passed")
        return relevant
    
    