"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / ".pageindex_cache.json"


def _load_cache() -> dict[str, str]:
    """Load cached document IDs to avoid re-uploading the same docs."""
    try:
        if not CACHE_PATH.exists():
            return {}
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    """Persist the document mapping to disk."""
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        return None

    try:
        import pageindex  # type: ignore
    except ModuleNotFoundError:
        return None

    cache = _load_cache()
    if not STANDARDIZED_DIR.exists():
        return None

    for file_path in sorted(STANDARDIZED_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        rel_name = str(file_path.relative_to(Path(__file__).parent.parent))
        if rel_name in cache:
            continue

        try:
            with file_path.open("rb") as handle:
                payload = handle.read()
            if not payload:
                continue
            client = getattr(pageindex, "PageIndexClient", None)
            if client is None:
                cache[rel_name] = f"local:{rel_name}"
                continue
            response = client(api_key=PAGEINDEX_API_KEY, timeout=15).upload_document(
                str(file_path),
                metadata={"source": rel_name},
            )
            doc_id = response
            if isinstance(response, dict):
                doc_id = (
                    response.get("id")
                    or response.get("document_id")
                    or response.get("data", {}).get("id")
                    or response.get("document", {}).get("id")
                    or str(rel_name)
                )
            cache[rel_name] = str(doc_id)
        except Exception:
            # Provider-specific behavior must never crash the pipeline.
            continue

    _save_cache(cache)
    return None


def _coerce_search_results(raw_response: object, *, top_k: int) -> list[dict]:
    """Normalize PageIndex responses across a few common SDK shapes."""
    if not isinstance(raw_response, dict):
        return []

    candidates: list[object] = []
    for key in ("results", "data", "items", "documents", "chunks"):
        value = raw_response.get(key)
        if isinstance(value, list):
            candidates = value
            break
    if not candidates:
        candidates = [raw_response]

    normalized: list[dict] = []
    for index, item in enumerate(candidates[: max(top_k, 1)]):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("id") or item.get("document_id") or item.get("chunk_id") or f"pageindex-{index}"
        content = item.get("content") or item.get("text") or item.get("snippet") or ""
        score = item.get("score")
        if score is None:
            score = item.get("relevance")
        if score is None:
            score = item.get("similarity")
        if not isinstance(score, (int, float)):
            score = float(max(top_k - index, 0))
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {
                "source": item.get("source", "pageindex"),
                "title": item.get("title", "PageIndex result"),
                "doc_type": item.get("doc_type", "legal"),
                "url": item.get("url"),
                "chunk_index": index,
            }
        normalized.append(
            {
                "id": str(doc_id),
                "content": str(content),
                "score": float(score),
                "metadata": metadata,
                "retrieval_method": "pageindex",
            }
        )

    normalized.sort(key=lambda entry: entry["score"], reverse=True)
    return normalized[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not PAGEINDEX_API_KEY:
        return []

    try:
        import pageindex  # type: ignore
    except ModuleNotFoundError:
        return []

    try:
        client = getattr(pageindex, "PageIndexClient", None)
        if client is None:
            return []
        response = client(api_key=PAGEINDEX_API_KEY, timeout=15).search(
            query=query,
            top_k=top_k,
        )
        return _coerce_search_results(response, top_k=top_k)
    except Exception:
        return []


if __name__ == "__main__":
    upload_documents()
