from ingest import get_collection, get_embedder
from config import TOP_K_RETRIEVE, TOP_K_RERANK, RERANK_MODEL

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model (downloads ~68 MB on first run)...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def contextualize_query(question: str, history: list[dict]) -> str:
    """Expand short follow-up questions using the last user question for better embedding search."""
    if not history or len(question.split()) > 8:
        return question
    last_user_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"{last_user_q} {question}" if last_user_q else question


def search(question: str, subject: str, book_id: str | None = None) -> list[dict]:
    collection = get_collection()
    embedder = get_embedder()

    query_vec = embedder.encode([question])[0].tolist()

    where = (
        {"$and": [{"subject": subject}, {"book_id": book_id}]}
        if book_id
        else {"subject": subject}
    )

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=TOP_K_RETRIEVE,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=1,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append(
            {
                "text": doc,
                "title": meta["title"],
                "author": meta["author"],
                "page": meta["page"],
                "book_id": meta["book_id"],
                "has_visuals": bool(meta.get("has_visuals", False)),
                "embed_score": round(1 - dist, 3),
            }
        )

    if not candidates:
        return []

    # Rerank: score all candidates against the question, keep the best TOP_K_RERANK.
    reranker = get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for chunk, score in zip(candidates, scores):
        chunk["score"] = round(float(score), 3)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:TOP_K_RERANK]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f'[Excerpt {i} — "{c["title"]}" by {c["author"]}, Page {c["page"]}]\n{c["text"]}'
        )
    return "\n\n---\n\n".join(parts)


def format_sources(chunks: list[dict]) -> str:
    seen, lines = set(), []
    for c in chunks:
        key = f"{c['title']}|{c['page']}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  • \"{c['title']}\" by {c['author']}  —  Page {c['page']}")
    return "\n".join(lines)
