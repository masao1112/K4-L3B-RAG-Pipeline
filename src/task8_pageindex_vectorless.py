"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
ROOT_DIR = Path(__file__).parent.parent
CACHE_PATH = ROOT_DIR / "pageindex_doc_ids.json"
PDF_CACHE_DIR = ROOT_DIR / "pageindex_pdfs"


def _api_key() -> str:
    return os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        cache: dict[str, dict] = {}
        for source, value in raw.items():
            if isinstance(value, str):
                cache[source] = {"doc_id": value, "source": source}
            elif isinstance(value, dict) and value.get("doc_id"):
                cache[source] = value
        return cache
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    temporary_path = CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(CACHE_PATH)


def _document_metadata(path: Path, content: str) -> dict:
    title_match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    url_match = re.search(
        r"^\*\*Source:\*\*\s*(https?://\S+)",
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return {
        "source": path.name,
        "title": title_match.group(1).strip() if title_match else path.stem,
        "doc_type": "legal" if "legal" in path.parts else "news",
        "url": url_match.group(1).rstrip(")]}.,") if url_match else None,
    }


def _to_pdf(path: Path, content: str) -> Path:
    """Return a PDF accepted by PageIndex, reusing source PDFs when possible."""
    landing_pdf = ROOT_DIR / "data" / "landing" / "legal" / f"{path.stem}.pdf"
    if landing_pdf.exists():
        return landing_pdf

    target = PDF_CACHE_DIR / f"{path.parent.name}-{path.stem}.pdf"
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return target

    from fpdf import FPDF

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.exists():
        pdf.add_font("DejaVu", fname=str(font_path))
        pdf.set_font("DejaVu", size=10)
    else:
        pdf.set_font("Helvetica", size=10)
        content = content.encode("latin-1", errors="replace").decode("latin-1")

    for line in content.splitlines():
        # Character-sized segments prevent very long URLs from exceeding width.
        segments = [line[index:index + 100] for index in range(0, len(line), 100)]
        for segment in segments or [""]:
            pdf.multi_cell(0, 5, segment)
    pdf.output(str(target))
    return target


def _find_result_items(payload: object) -> list[dict]:
    """Find retrieval nodes across PageIndex response variants."""
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            if any(
                any(key in item for key in ("content", "text", "node_text", "summary"))
                for item in payload
            ):
                return payload
        for value in payload:
            found = _find_result_items(value)
            if found:
                return found
    elif isinstance(payload, dict):
        for key in ("results", "retrieved_nodes", "nodes", "passages", "data", "result"):
            if key in payload:
                found = _find_result_items(payload[key])
                if found:
                    return found
        for value in payload.values():
            found = _find_result_items(value)
            if found:
                return found
    return []


def _node_content(node: dict) -> str:
    for key in ("content", "text", "node_text", "summary"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    api_key = _api_key()
    if not api_key:
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    cache = _load_cache()
    changed = False
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name in cache:
            continue
        try:
            content = path.read_text(encoding="utf-8")
            upload_path = _to_pdf(path, content)
            response = client.submit_document(str(upload_path))
            doc_id = response.get("doc_id") if isinstance(response, dict) else None
            if not isinstance(doc_id, str) or not doc_id:
                continue
            cache[path.name] = {
                "doc_id": doc_id,
                **_document_metadata(path, content),
            }
            changed = True
        except Exception:
            # One bad document must not prevent other documents from uploading.
            continue
    if changed:
        _save_cache(cache)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    try:
        api_key = _api_key()
        if not api_key or top_k <= 0 or not query.strip():
            return []

        from pageindex import PageIndexClient

        cache = _load_cache()
        if not cache:
            upload_documents()
            cache = _load_cache()
        if not cache:
            return []

        client = PageIndexClient(api_key=api_key)
        results: list[dict] = []
        deadline = time.monotonic() + 20.0
        for source, cached in cache.items():
            if time.monotonic() >= deadline:
                break
            doc_id = cached.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            if hasattr(client, "is_retrieval_ready") and not client.is_retrieval_ready(doc_id):
                continue

            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id", "")
            if not retrieval_id:
                continue

            payload: object = {}
            while time.monotonic() < deadline:
                payload = client.get_retrieval(retrieval_id)
                nodes = _find_result_items(payload)
                if nodes:
                    break
                status = payload.get("status", "") if isinstance(payload, dict) else ""
                if str(status).lower() in {"failed", "error", "cancelled"}:
                    break
                if str(status).lower() in {"completed", "complete", "success", "ready"}:
                    break
                time.sleep(0.5)

            for node_index, node in enumerate(_find_result_items(payload)):
                content = _node_content(node)
                if not content:
                    continue
                raw_score = node.get("score") or node.get("relevance_score")
                score = float(raw_score) if raw_score is not None else 1.0 / (node_index + 1)
                try:
                    chunk_index = int(node.get("chunk_index", node_index))
                except (TypeError, ValueError):
                    chunk_index = node_index
                metadata = {
                    "source": cached.get("source") or source,
                    "title": cached.get("title") or Path(source).stem,
                    "doc_type": cached.get("doc_type") or "news",
                    "url": cached.get("url"),
                    "chunk_index": chunk_index,
                }
                node_id = node.get("id") or node.get("node_id") or node.get("nodeId")
                results.append(
                    {
                        "id": f"pageindex::{doc_id}::{node_id or node_index}",
                        "content": content,
                        "score": score,
                        "metadata": metadata,
                        "retrieval_method": "pageindex",
                    }
                )

        unique_results: dict[str, dict] = {}
        for result in sorted(results, key=lambda item: item["score"], reverse=True):
            unique_results.setdefault(result["id"], result)
        return list(unique_results.values())[:top_k]
    except Exception:
        return []


if __name__ == "__main__":
    upload_documents()
