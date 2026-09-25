"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


from .task4_chunking_indexing import chunk_documents, load_documents


# BM25 must search the same chunks that are written to Chroma by Task 4.
CORPUS: list[dict] = chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [item["content"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)

    # With exactly two documents, Okapi's IDF is mathematically zero for a
    # term occurring in one document. A tiny positive value preserves useful
    # exact matches while leaving normal corpora/rankings unchanged.
    for token, idf in bm25.idf.items():
        if idf == 0.0:
            bm25.idf[token] = 1e-12
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip() or not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(query.lower().split())
    ranked_indices = sorted(
        range(len(CORPUS)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results: list[dict] = []
    seen_ids: set[str] = set()
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        item = CORPUS[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
