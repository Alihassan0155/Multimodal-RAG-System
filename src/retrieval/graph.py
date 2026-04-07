# src/retrieval/graph.py

import re
from collections import defaultdict
from math import log

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 CONCEPT: RAG Graph
#
# A graph where nodes are chunks and edges represent relationships.
# After initial retrieval, the graph EXPANDS the context by pulling
# in related chunks that the retriever may have missed.
#
# WHY GRAPHS IMPROVE RAG:
# Standard retrieval finds chunks most similar to the query.
# But the ANSWER often requires chunks that are:
#   - Adjacent to retrieved chunks (sequential context)
#   - From the same section (topical context)
#   - Sharing key concepts (semantic bridges)
#
# Example without graph:
#   Query: "how does the GNN model predict resistance?"
#   Retrieved: chunk about GNN architecture (page 2)
#   Missing: chunk about prediction output (page 2, next chunk)
#             → answer is incomplete
#
# Example with graph:
#   Retrieved: GNN architecture chunk
#   Graph expands → pulls in sequential neighbour (prediction chunk)
#   → complete answer
#
# THREE EDGE TYPES (weighted):
#
# SEQUENTIAL (weight 0.8)
#   chunk_i → chunk_i+1 (document order)
#   Strongest signal — author wrote these to be read together.
#   If chunk A is relevant, the chunk immediately after is very likely
#   to contain completing information.
#
# SAME_SECTION (weight 0.6)
#   All chunks from same page share an edge.
#   Medium signal — same page = same topic cluster usually.
#   Weaker than sequential because page can contain multiple unrelated sections.
#
# KEYWORD_OVERLAP (weight 0.4)
#   Chunks sharing significant keywords (TF-IDF weighted).
#   Weakest signal — keyword co-occurrence suggests related concepts
#   but doesn't guarantee answer-level relevance.
#   KEY USE CASE: bridges text chunks ↔ table chunks.
#   "plasmid" mentioned in text → "Plasmid" row in table chunk.
#
# WHERE ELSE ARE KNOWLEDGE GRAPHS USED IN RAG:
#   Microsoft GraphRAG, LlamaIndex Knowledge Graph Index,
#   NebulaGraph + LLM pipelines. Graph expansion is one of the most
#   active research areas in RAG improvement (2024-2025).
# ─────────────────────────────────────────────────────────────────────────────

# Edge type weights — controls priority during graph expansion
EDGE_WEIGHTS = {
    "sequential":       0.8,
    "same_section":     0.6,
    "keyword_overlap":  0.4,
}

# Minimum keyword overlap score to create an edge
KEYWORD_OVERLAP_THRESHOLD = 0.10

# Maximum neighbours to add per retrieved chunk during expansion
MAX_NEIGHBOURS_PER_CHUNK = 2

# Stopwords to exclude from keyword overlap calculation
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "its", "we", "our", "they", "their",
}


class ChunkGraph:
    """
    Graph of chunk relationships built at ingest time.
    Used at query time to expand retrieved chunks with neighbours.

    Build once per session:
        graph = ChunkGraph(all_chunks)

    Expand at query time:
        expanded = graph.expand(retrieved_chunks, top_n=3)
    """

    def __init__(self, chunks: list[dict]):
        """
        Builds the full chunk graph from all chunks in the corpus.

        Args:
            chunks: list of all chunk dicts from ChromaDB
        """
        self.chunks    = chunks
        self.id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

        # Adjacency list: chunk_id → list of (neighbour_id, edge_type, weight)
        self.edges: dict[str, list[tuple[str, str, float]]] = defaultdict(list)

        print("  [Graph] Building chunk relationship graph...")
        self._build_sequential_edges()
        self._build_same_section_edges()
        self._build_keyword_overlap_edges()

        total_edges = sum(len(v) for v in self.edges.values())
        print(f"  [Graph] Graph built: {len(chunks)} nodes, {total_edges} edges")


    # ── Edge builders ─────────────────────────────────────────────────────────

    def _build_sequential_edges(self):
        """
        Connects adjacent chunks in document order.

        🧠 Sort by (source, page, chunk_index) to get document order.
        Then connect each chunk to the next one.
        Edges are bidirectional — chunk_i ↔ chunk_i+1.

        Why bidirectional:
        If the retrieved chunk is chunk_5, the user might need chunk_4
        (what came before) OR chunk_6 (what came after).
        Both directions are potentially useful context.
        """
        # Sort chunks into document reading order
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: (
                c.get("source", ""),
                c.get("page", 0),
                c.get("chunk_index", 0),
            )
        )

        weight = EDGE_WEIGHTS["sequential"]

        for i in range(len(sorted_chunks) - 1):
            curr = sorted_chunks[i]
            nxt  = sorted_chunks[i + 1]

            # Only connect chunks from the same document
            if curr.get("source") != nxt.get("source"):
                continue

            self.edges[curr["id"]].append((nxt["id"],  "sequential", weight))
            self.edges[nxt["id"]].append((curr["id"], "sequential", weight))

        print(f"  [Graph] Sequential edges built")


    def _build_same_section_edges(self):
        """
        Connects all chunks from the same page.

        🧠 Groups chunks by (source, page).
        All chunks within a group share edges with each other.
        This is O(k²) per page where k = chunks per page.
        For typical PDFs (2-5 chunks per page) this is trivial.
        """
        # Group by (source, page)
        page_groups: dict[tuple, list[dict]] = defaultdict(list)
        for chunk in self.chunks:
            key = (chunk.get("source", ""), chunk.get("page", 0))
            page_groups[key].append(chunk)

        weight = EDGE_WEIGHTS["same_section"]

        for group in page_groups.values():
            if len(group) < 2:
                continue  # single chunk on page — no same_section edges

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    # Skip if already connected by sequential edge
                    existing = {e[0] for e in self.edges[a["id"]]}
                    if b["id"] not in existing:
                        self.edges[a["id"]].append((b["id"], "same_section", weight))
                        self.edges[b["id"]].append((a["id"], "same_section", weight))

        print(f"  [Graph] Same-section edges built")


    def _build_keyword_overlap_edges(self):
        """
        Connects chunks that share significant keywords.

        🧠 CONCEPT: TF-IDF weighted keyword overlap
        We can't just count shared words — "the", "is", "of" appear
        everywhere and would create edges between unrelated chunks.

        Instead we weight each word by its IDF score:
            IDF(word) = log(N / df)
        Where N = total chunks, df = chunks containing this word.

        Rare words ("plasmid", "GNN", "FASTA") get high IDF → high signal.
        Common words ("the", "system") get low IDF → low signal.

        Overlap score between chunk A and chunk B:
            score = Σ IDF(shared_word) / max(|words_A|, |words_B|)

        If score > KEYWORD_OVERLAP_THRESHOLD → create edge.

        WHY THIS IS THE MOST VALUABLE EDGE FOR MULTIMODAL RAG:
        Text chunk mentions "plasmid" → table chunk defines "Plasmid".
        Without this edge, text retrieval never surfaces the table definition.
        With this edge, retrieving the text chunk automatically pulls
        in the table chunk as a neighbour → richer answer.
        """
        # Tokenize all chunks
        def tokenize(text: str) -> set[str]:
            words = re.findall(r'\b[a-z]{2,}\b', text.lower())
            return {w for w in words if w not in STOPWORDS}

        chunk_tokens = [tokenize(c.get("content", "")) for c in self.chunks]

        # Compute IDF for each word
        N  = len(self.chunks)
        df = defaultdict(int)
        for tokens in chunk_tokens:
            for word in tokens:
                df[word] += 1

        idf = {word: log(N / count) for word, count in df.items() if count < N}

        weight = EDGE_WEIGHTS["keyword_overlap"]
        edges_created = 0

        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):
                tokens_i = chunk_tokens[i]
                tokens_j = chunk_tokens[j]

                shared = tokens_i & tokens_j
                if not shared:
                    continue

                # Weighted overlap score
                overlap_score = (
                    sum(idf.get(w, 0) for w in shared)
                    / max(len(tokens_i), len(tokens_j), 1)
                )

                if overlap_score >= KEYWORD_OVERLAP_THRESHOLD:
                    a = self.chunks[i]
                    b = self.chunks[j]

                    # Don't duplicate same_section edges
                    existing = {e[0] for e in self.edges[a["id"]]}
                    if b["id"] not in existing:
                        self.edges[a["id"]].append((b["id"], "keyword_overlap", weight))
                        self.edges[b["id"]].append((a["id"], "keyword_overlap", weight))
                        edges_created += 1

        print(f"  [Graph] Keyword overlap edges built ({edges_created} edges)")


    # ── Query-time expansion ──────────────────────────────────────────────────

    def expand(
        self,
        retrieved_chunks: list[dict],
        top_n:            int = 3,
    ) -> list[dict]:
        """
        Expands a set of retrieved chunks with their graph neighbours.

        🧠 HOW EXPANSION WORKS:
        For each retrieved chunk:
          1. Look up its edges in the graph
          2. Sort neighbours by edge weight (sequential first)
          3. Add top-N neighbours not already in retrieved set
          4. Deduplicate across all additions

        The expanded set goes to the reranker which scores all chunks
        against the original query — so irrelevant graph additions
        get scored low and naturally filtered out.

        Args:
            retrieved_chunks : initial retrieval results
            top_n            : max neighbours to add per retrieved chunk

        Returns:
            Original chunks + graph neighbours (deduplicated)
        """
        if not retrieved_chunks:
            return retrieved_chunks

        # Build lookup of already-retrieved IDs
        id_to_chunk  = {c["id"]: c for c in self.chunks}
        retrieved_ids = {c["id"] for c in retrieved_chunks}
        additions     = []

        print(f"  [Graph] Expanding {len(retrieved_chunks)} chunks...")

        for chunk in retrieved_chunks:
            chunk_id   = chunk["id"]
            neighbours = self.edges.get(chunk_id, [])

            if not neighbours:
                continue

            # Sort by weight descending (sequential first)
            neighbours_sorted = sorted(
                neighbours,
                key=lambda x: x[2],
                reverse=True,
            )

            added = 0
            for neighbour_id, edge_type, edge_weight in neighbours_sorted:
                if added >= top_n:
                    break

                # Skip if already retrieved or already being added
                if neighbour_id in retrieved_ids:
                    continue
                if neighbour_id in {a["id"] for a in additions}:
                    continue

                neighbour_chunk = id_to_chunk.get(neighbour_id)
                if not neighbour_chunk:
                    continue

                # Tag with graph metadata so we can see what was added
                enriched = neighbour_chunk.copy()
                enriched["graph_edge"]        = edge_type
                enriched["graph_edge_weight"] = edge_weight
                enriched["graph_source"]      = chunk_id

                additions.append(enriched)
                added += 1

                print(f"  [Graph] Added neighbour via {edge_type} "
                      f"(weight {edge_weight}): {neighbour_id}")

        expanded = retrieved_chunks + additions
        print(f"  [Graph] Expansion: {len(retrieved_chunks)} → {len(expanded)} chunks")
        return expanded